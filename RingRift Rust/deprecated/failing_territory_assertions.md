# Failing Assertions in `ringrift/src/rules/territory.rs`

Based on the test output from `cargo test --manifest-path ringrift/Cargo.toml --lib rules::territory` run at 2025-04-22 10:33:58 UTC-05:00:

1.  **`test_scenario_position_2`** (panicked at `src/rules/territory.rs:655:9`)
    *   **Failing Assertion:**
        ```rust
        assert!(fully_disconnected_regions.iter().any(|(r, c)| r.len() == 8 && *c == PlayerColor::Red), "Did not find Inner-R fully disconnected region (size 8)");
        ```
    *   **Reason:** The function did not return a fully disconnected region of size 8 bounded by Red.

2.  **`test_is_fully_disconnected_region_scenarios`** (panicked at `src/rules/territory.rs:481:9`)
    *   **Failing Assertion:**
        ```rust
        assert_eq!(fully1.len(), 2, "Scenario 1: Expected 2 fully disconnected regions (Inner-B, Outer-B)");
        ```
    *   **Reason:** The function returned 3 regions instead of the expected 2 for Scenario 1.

3.  **`test_scenario_position_2_remove_markers`** (panicked at `src/rules/territory.rs:684:9`)
    *   **Failing Assertion:**
        ```rust
        assert_eq!(green_physical_count, 0, "Position 2 (no b5r): Expected 0 physically disconnected regions bounded by Green");
        ```
    *   **Reason:** In the "no b5r" sub-case, the function returned 2 physically disconnected regions bounded by Green, but 0 were expected.
