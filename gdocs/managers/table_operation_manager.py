"""
Table Operation Manager

This module provides high-level table operations that orchestrate
multiple Google Docs API calls for complex table manipulations.

Uses batch operations to minimize API calls and avoid rate limiting.
A table creation + population takes 3 API calls instead of N*M+1.
"""

import logging
import asyncio
from typing import List, Dict, Any, Tuple

from gdocs.docs_helpers import create_insert_table_request
from gdocs.docs_structure import find_tables
from gdocs.docs_tables import validate_table_data, build_table_population_requests

logger = logging.getLogger(__name__)


def _reverse_cell_groups(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reverse the order of cell groups for correct Google Docs batch execution.

    Each cell group starts with an insertText and may be followed by an
    updateTextStyle (bold). We reverse the GROUP order so cells are processed
    from highest index to lowest, but within each group the insertText still
    comes before the bold formatting.

    This is necessary because Google Docs batchUpdate applies requests
    sequentially — inserting text shifts all subsequent indices.
    """
    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []

    for req in requests:
        if "insertText" in req:
            # Start a new group
            if current_group:
                groups.append(current_group)
            current_group = [req]
        else:
            # updateTextStyle (bold) belongs to the current group
            current_group.append(req)

    if current_group:
        groups.append(current_group)

    # Reverse group order, flatten back
    groups.reverse()
    result: List[Dict[str, Any]] = []
    for group in groups:
        result.extend(group)
    return result


class TableOperationManager:
    """
    High-level manager for Google Docs table operations.

    Uses batch operations to minimize API calls:
    - create_and_populate_table: 3 API calls (create + read structure + batch populate)
    - populate_existing_table: 2 API calls (read structure + batch populate)
    """

    def __init__(self, service):
        """
        Initialize the table operation manager.

        Args:
            service: Google Docs API service instance
        """
        self.service = service

    async def create_and_populate_table(
        self,
        document_id: str,
        table_data: List[List[str]],
        index: int,
        bold_headers: bool = True,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Creates a table and populates it with data using batch operations.

        Uses 3 API calls total:
        1. batchUpdate with insertTable (create empty table)
        2. documents().get() (read structure to get cell indices)
        3. batchUpdate with all insertText + formatting ops (populate all cells at once)

        Args:
            document_id: ID of the document to update
            table_data: 2D list of strings for table content
            index: Position to insert the table
            bold_headers: Whether to make the first row bold

        Returns:
            Tuple of (success, message, metadata)
        """
        logger.debug(
            f"Creating table at index {index}, dimensions: "
            f"{len(table_data)}x{len(table_data[0]) if table_data and len(table_data) > 0 else 0}"
        )

        # Validate input data
        is_valid, error_msg = validate_table_data(table_data)
        if not is_valid:
            return False, f"Invalid table data: {error_msg}", {}

        rows = len(table_data)
        cols = len(table_data[0])

        try:
            # Step 1: Create empty table (1 API call)
            await self._create_empty_table(document_id, index, rows, cols)

            # Step 2: Get fresh document structure to find actual cell positions (1 API call)
            fresh_tables = await self._get_document_tables(document_id)
            if not fresh_tables:
                return False, "Could not find table after creation", {}

            # Use the last table (the one we just created)
            table_info = fresh_tables[-1]

            # Step 3: Build all cell population requests as a batch
            requests = build_table_population_requests(
                table_info, table_data, bold_headers
            )

            if not requests:
                # No cells to populate (all empty strings)
                metadata = {
                    "rows": rows,
                    "columns": cols,
                    "populated_cells": 0,
                    "table_index": len(fresh_tables) - 1,
                }
                return (
                    True,
                    f"Successfully created {rows}x{cols} table (no cell data to populate)",
                    metadata,
                )

            # CRITICAL: Reverse CELL GROUP order for correct batch execution.
            # Google Docs batchUpdate processes requests sequentially — each insertText
            # shifts indices after it. By processing cells from highest index to lowest,
            # earlier insertions don't invalidate later cell indices.
            # BUT within each cell group, insertText must come before updateTextStyle
            # (bold), so we reverse groups, not individual requests.
            requests = _reverse_cell_groups(requests)

            # Step 4: Execute all cell insertions in a single batchUpdate (1 API call)
            result = await asyncio.to_thread(
                self.service.documents()
                .batchUpdate(
                    documentId=document_id,
                    body={"requests": requests},
                )
                .execute
            )

            replies = result.get("replies", [])
            # Count successful insertText operations (non-empty replies)
            populated_cells = sum(
                1 for r in replies if r  # Non-empty reply = successful operation
            )
            # If all replies are empty dicts (normal for successful insertText),
            # count the insertText requests
            if populated_cells == 0:
                populated_cells = sum(
                    1 for r in requests if "insertText" in r
                )

            metadata = {
                "rows": rows,
                "columns": cols,
                "populated_cells": populated_cells,
                "table_index": len(fresh_tables) - 1,
                "api_calls": 3,
                "batch_requests": len(requests),
            }

            logger.info(
                f"Table {rows}x{cols} created and populated with {populated_cells} cells "
                f"using 3 API calls (batch of {len(requests)} operations)"
            )

            return (
                True,
                f"Successfully created {rows}x{cols} table and populated {populated_cells} cells",
                metadata,
            )

        except Exception as e:
            logger.error(f"Failed to create and populate table: {str(e)}")
            return False, f"Table creation failed: {str(e)}", {}

    async def _create_empty_table(
        self, document_id: str, index: int, rows: int, cols: int
    ) -> None:
        """Create an empty table at the specified index."""
        logger.debug(f"Creating {rows}x{cols} table at index {index}")

        await asyncio.to_thread(
            self.service.documents()
            .batchUpdate(
                documentId=document_id,
                body={"requests": [create_insert_table_request(index, rows, cols)]},
            )
            .execute
        )

    async def _get_document_tables(self, document_id: str) -> List[Dict[str, Any]]:
        """Get fresh document structure and extract table information."""
        doc = await asyncio.to_thread(
            self.service.documents().get(documentId=document_id).execute
        )
        return find_tables(doc)

    async def populate_existing_table(
        self,
        document_id: str,
        table_index: int,
        table_data: List[List[str]],
        clear_existing: bool = False,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Populate an existing table with data using batch operations.

        Uses 2 API calls:
        1. documents().get() (read structure to get cell indices)
        2. batchUpdate with all insertText ops (populate all cells at once)

        Args:
            document_id: ID of the document
            table_index: Index of the table to populate (0-based)
            table_data: 2D list of data to insert
            clear_existing: Whether to clear existing content first

        Returns:
            Tuple of (success, message, metadata)
        """
        try:
            tables = await self._get_document_tables(document_id)
            if table_index >= len(tables):
                return (
                    False,
                    f"Table index {table_index} not found. Document has {len(tables)} tables",
                    {},
                )

            table_info = tables[table_index]

            # Validate dimensions
            table_rows = table_info["rows"]
            table_cols = table_info["columns"]
            data_rows = len(table_data)
            data_cols = len(table_data[0]) if table_data else 0

            if data_rows > table_rows or data_cols > table_cols:
                return (
                    False,
                    f"Data ({data_rows}x{data_cols}) exceeds table dimensions ({table_rows}x{table_cols})",
                    {},
                )

            # Build batch requests for all cells
            requests = build_table_population_requests(
                table_info, table_data, bold_headers=False
            )

            if not requests:
                return (
                    True,
                    "No cell data to populate",
                    {"table_index": table_index, "populated_cells": 0},
                )

            # Reverse cell group order for correct batch execution (end-to-start)
            requests = _reverse_cell_groups(requests)

            # Execute all cell insertions in a single batchUpdate
            await asyncio.to_thread(
                self.service.documents()
                .batchUpdate(
                    documentId=document_id,
                    body={"requests": requests},
                )
                .execute
            )

            population_count = sum(1 for r in requests if "insertText" in r)

            metadata = {
                "table_index": table_index,
                "populated_cells": population_count,
                "table_dimensions": f"{table_rows}x{table_cols}",
                "data_dimensions": f"{data_rows}x{data_cols}",
                "api_calls": 2,
            }

            return (
                True,
                f"Successfully populated {population_count} cells in existing table",
                metadata,
            )

        except Exception as e:
            return False, f"Failed to populate existing table: {str(e)}", {}
