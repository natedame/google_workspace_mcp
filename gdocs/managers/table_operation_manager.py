"""
Table Operation Manager

This module provides high-level table operations that orchestrate
multiple Google Docs API calls for complex table manipulations.

Uses batch operations to minimize API calls and avoid rate limiting.
A table creation + population takes 5 API calls (snapshot + create + read + populate + verify).
Identifies newly created tables via before/after diffing to prevent wrong-table population.
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
    - create_and_populate_table: 5 API calls (snapshot + create + read + batch populate + verify)
    - populate_existing_table: 2 API calls (read structure + batch populate)
    """

    def __init__(self, service):
        """
        Initialize the table operation manager.

        Args:
            service: Google Docs API service instance
        """
        self.service = service

    def _find_new_table(
        self,
        existing_tables: List[Dict[str, Any]],
        fresh_tables: List[Dict[str, Any]],
        expected_rows: int,
        expected_cols: int,
        insertion_index: int,
    ) -> Dict[str, Any] | None:
        """
        Identify the newly created table by diffing before/after table lists.

        Compares sorted table lists to find the extra entry that appeared after
        creation. Handles index shifting (tables after the insertion point will
        have shifted start_indices).

        Args:
            existing_tables: Tables before creation (sorted by start_index)
            fresh_tables: Tables after creation (sorted by start_index)
            expected_rows: Expected row count of the new table
            expected_cols: Expected column count of the new table
            insertion_index: Where we asked the API to insert the table

        Returns:
            The new table's info dict, or None if not found
        """
        existing_sorted = sorted(existing_tables, key=lambda t: t["start_index"])
        fresh_sorted = sorted(fresh_tables, key=lambda t: t["start_index"])

        if len(fresh_sorted) != len(existing_sorted) + 1:
            logger.error(
                f"Expected {len(existing_sorted) + 1} tables after creation, "
                f"found {len(fresh_sorted)}"
            )
            return None

        # Walk both lists to find the insertion point.
        # The new table is the entry in fresh_sorted that has no corresponding
        # entry in existing_sorted (accounting for index shift).
        j = 0  # pointer into existing_sorted
        for i, ft in enumerate(fresh_sorted):
            if j < len(existing_sorted):
                et = existing_sorted[j]
                # Match by dimensions + emptiness: an existing (already-populated)
                # table will have the same rows/cols and non-empty cells.
                # The new table will be empty and at or near the insertion index.
                if ft["rows"] == et["rows"] and ft["columns"] == et["columns"]:
                    # Check if this fresh table has content (matches existing populated table)
                    ft_has_content = any(
                        cell.get("content", "").strip()
                        for row in ft.get("cells", [])
                        for cell in row
                    )
                    if ft_has_content:
                        # This matches an existing table, skip
                        j += 1
                        continue

            # This table in fresh_sorted doesn't match the next expected existing table.
            # Verify it's the right dimensions and is empty.
            if ft["rows"] == expected_rows and ft["columns"] == expected_cols:
                all_empty = all(
                    not cell.get("content", "").strip()
                    for row in ft.get("cells", [])
                    for cell in row
                )
                if all_empty:
                    logger.info(
                        f"Identified new {expected_rows}x{expected_cols} table at "
                        f"index {ft['start_index']} via before/after diff "
                        f"(insertion was {insertion_index})"
                    )
                    return ft

            # Doesn't match — keep walking
            j += 1

        # Fallback: the new table might be at the end
        if fresh_sorted and len(fresh_sorted) > len(existing_sorted):
            last = fresh_sorted[-1]
            if last["rows"] == expected_rows and last["columns"] == expected_cols:
                all_empty = all(
                    not cell.get("content", "").strip()
                    for row in last.get("cells", [])
                    for cell in row
                )
                if all_empty:
                    logger.info(
                        f"Identified new table at end: index {last['start_index']} "
                        f"(insertion was {insertion_index})"
                    )
                    return last

        return None

    async def _verify_table_population(
        self,
        document_id: str,
        table_start_index: int,
        table_data: List[List[str]],
        rows: int,
        cols: int,
    ) -> Tuple[bool, str]:
        """
        Verify that table cells were actually populated with expected data.

        Reads the document again and checks that the first row's cells
        contain the expected text.

        Returns:
            Tuple of (verified, message)
        """
        try:
            verify_tables = await self._get_document_tables(document_id)
            # Find the table by start_index proximity (it may have shifted slightly)
            target = min(
                verify_tables,
                key=lambda t: abs(t["start_index"] - table_start_index),
            )

            if target["rows"] != rows or target["columns"] != cols:
                return False, (
                    f"Verification: table dimensions mismatch "
                    f"(expected {rows}x{cols}, got {target['rows']}x{target['columns']})"
                )

            # Check first row cells contain expected data
            errors = []
            for col_idx, expected_text in enumerate(table_data[0]):
                if col_idx >= len(target["cells"][0]):
                    break
                actual = target["cells"][0][col_idx].get("content", "").strip()
                if expected_text and expected_text not in actual:
                    errors.append(
                        f"cell(0,{col_idx}): expected '{expected_text[:30]}', "
                        f"got '{actual[:30]}'"
                    )

            if errors:
                return False, f"Verification failed: {'; '.join(errors)}"

            # Spot-check a middle row if table has more than 2 rows
            if len(table_data) > 2:
                mid = len(table_data) // 2
                if mid < len(target["cells"]):
                    for col_idx, expected_text in enumerate(table_data[mid]):
                        if col_idx >= len(target["cells"][mid]):
                            break
                        actual = target["cells"][mid][col_idx].get("content", "").strip()
                        if expected_text and expected_text not in actual:
                            errors.append(
                                f"cell({mid},{col_idx}): expected '{expected_text[:30]}', "
                                f"got '{actual[:30]}'"
                            )

            if errors:
                return False, f"Verification failed: {'; '.join(errors)}"

            return True, "Verification passed"

        except Exception as e:
            logger.warning(f"Verification read failed: {e}")
            return False, f"Verification read failed: {e}"

    async def create_and_populate_table(
        self,
        document_id: str,
        table_data: List[List[str]],
        index: int,
        bold_headers: bool = True,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Creates a table and populates it with data using batch operations.

        Uses 5 API calls total:
        1. documents().get() (snapshot existing tables for diff)
        2. batchUpdate with insertTable (create empty table)
        3. documents().get() (read structure to get cell indices)
        4. batchUpdate with all insertText + formatting ops (populate all cells at once)
        5. documents().get() (verify cell population)

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
            # Step 1: Snapshot existing tables BEFORE creation (1 API call)
            existing_tables = await self._get_document_tables(document_id)
            logger.debug(
                f"Pre-creation snapshot: {len(existing_tables)} existing tables "
                f"at indices {[t['start_index'] for t in existing_tables]}"
            )

            # Step 2: Create empty table (1 API call)
            await self._create_empty_table(document_id, index, rows, cols)

            # Step 3: Get fresh document structure to find actual cell positions (1 API call)
            fresh_tables = await self._get_document_tables(document_id)
            if not fresh_tables:
                return False, "Could not find any tables after creation", {}

            # Identify the NEW table by diffing before/after lists.
            # This is far more reliable than the old heuristic of "find empty table
            # with matching dimensions" which fails when multiple same-dimension
            # tables exist (e.g., a previous population failed silently).
            table_info = self._find_new_table(
                existing_tables, fresh_tables, rows, cols, index
            )

            if table_info is None:
                # Diff-based identification failed. Try empty-table matching as fallback,
                # but ONLY consider tables with correct dimensions.
                logger.warning(
                    "Before/after diff did not identify new table, "
                    "falling back to empty-table matching"
                )
                empty_candidates = []
                for t in fresh_tables:
                    if t["rows"] != rows or t["columns"] != cols:
                        continue
                    all_empty = all(
                        not cell.get("content", "").strip()
                        for row in t.get("cells", [])
                        for cell in row
                    )
                    if all_empty:
                        empty_candidates.append(t)

                if len(empty_candidates) == 1:
                    table_info = empty_candidates[0]
                    logger.info(
                        f"Fallback: found 1 empty {rows}x{cols} table at "
                        f"{table_info['start_index']}"
                    )
                elif len(empty_candidates) > 1:
                    table_info = min(
                        empty_candidates,
                        key=lambda t: abs(t["start_index"] - index),
                    )
                    logger.warning(
                        f"Fallback: {len(empty_candidates)} empty {rows}x{cols} tables, "
                        f"using closest to index {index} at {table_info['start_index']}"
                    )
                else:
                    # FAIL EXPLICITLY instead of picking a random table.
                    # The old code had a dangerous fallback that ignored dimensions
                    # entirely — this caused data to be written to wrong tables.
                    logger.error(
                        f"FATAL: No empty {rows}x{cols} table found after creation! "
                        f"Existing tables: {[(t['start_index'], t['rows'], t['columns']) for t in fresh_tables]}. "
                        f"Refusing to populate a wrong table."
                    )
                    return (
                        False,
                        f"Table {rows}x{cols} was created at index {index} but could not be "
                        f"identified in the document. Found {len(fresh_tables)} tables, "
                        f"none empty with matching dimensions. This prevents data "
                        f"from being written to the wrong table.",
                        {},
                    )

            # Step 4: Build all cell population requests as a batch
            requests = build_table_population_requests(
                table_info, table_data, bold_headers
            )

            if not requests:
                metadata = {
                    "rows": rows,
                    "columns": cols,
                    "populated_cells": 0,
                    "table_index": fresh_tables.index(table_info) if table_info in fresh_tables else -1,
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

            # Step 5: Execute all cell insertions in a single batchUpdate (1 API call)
            result = await asyncio.to_thread(
                self.service.documents()
                .batchUpdate(
                    documentId=document_id,
                    body={"requests": requests},
                )
                .execute
            )

            replies = result.get("replies", [])
            populated_cells = sum(
                1 for r in replies if r
            )
            if populated_cells == 0:
                populated_cells = sum(
                    1 for r in requests if "insertText" in r
                )

            # Step 6: Verify cells were actually populated (1 API call)
            verified, verify_msg = await self._verify_table_population(
                document_id, table_info["start_index"], table_data, rows, cols
            )
            if not verified:
                logger.error(f"Post-population verification failed: {verify_msg}")
                return (
                    False,
                    f"Table {rows}x{cols} created and batch populate sent, but {verify_msg}",
                    {"rows": rows, "columns": cols, "verification_failed": True},
                )

            metadata = {
                "rows": rows,
                "columns": cols,
                "populated_cells": populated_cells,
                "table_index": len(fresh_tables) - 1,
                "api_calls": 5,
                "batch_requests": len(requests),
                "verified": True,
            }

            logger.info(
                f"Table {rows}x{cols} created, populated ({populated_cells} cells), "
                f"and verified using 5 API calls (batch of {len(requests)} operations)"
            )

            return (
                True,
                f"Successfully created {rows}x{cols} table, populated {populated_cells} cells (verified)",
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
