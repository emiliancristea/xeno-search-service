# Process Log: Phase 3 - Frontend Integration (`ChatWithLLM.tsx`)

**Overall Phase Goal:** Integrate the Xeno Search functionality into the `ChatWithLLM.tsx` frontend component. This will allow users to trigger a Xeno Search, view its results, and have that information automatically used as context for the LLM's response.

---

**Current Date:** May 18, 2025

## Step 1: Add UI Element for Xeno Search Activation

*   **File to Tweak:** `ChatWithLLM.tsx` (or related UI components).
*   **Action:** Implement a UI element (e.g., a toggle switch, a checkbox, or a button) that allows the user to enable/disable Xeno Search mode for their query.
*   **State Management:** Add a new state variable (e.g., `isXenoSearchEnabled`) to `ChatWithLLM.tsx` to track the status of this UI element.
*   **Rationale:** Provides user control over when to use the Xeno Search feature.
*   **Status:** Pending.

## Step 2: Modify `handleGenerate` (or Message Sending Logic)

*   **File to Tweak:** `ChatWithLLM.tsx`.
*   **Action:** Update the function responsible for sending a message/generating a response (e.g., `handleGenerate`):
    *   If `isXenoSearchEnabled` is true:
        *   First, make an API call to the Node.js backend endpoint `/api/xeno-search` with the user's query, `search_type: "normal"`, and a default `num_results`.
        *   Display a loading indicator to the user (e.g., "Performing Xeno Search...").
        *   Await the response from `/api/xeno-search`.
    *   If the call is successful and returns sources:
        *   Store the search results (summary, sources) in a new state variable (e.g., `xenoSearchResults`).
        *   Construct an augmented prompt for the LLM. This prompt should include the user's original query along with the key information retrieved by Xeno Search (e.g., the summary, or a compilation of snippets).
        *   Proceed to call the LLM with this augmented prompt.
    *   If Xeno Search is disabled, or if it fails or returns no useful results, proceed with the normal LLM call using just the user's query.
*   **Rationale:** Implements the core logic for conditionally invoking Xeno Search and using its output.
*   **Status:** Pending.

## Step 3: Display Xeno Search Results in UI

*   **File to Tweak:** `ChatWithLLM.tsx` (and potentially new sub-components).
*   **Action:** When `xenoSearchResults` state contains data:
    *   Render the search results in the chat interface before or alongside the LLM's response.
    *   Display the summary (if available).
    *   List the sources (title, URL, possibly a snippet). URLs should be clickable.
    *   Consider how to clearly distinguish Xeno Search results from the LLM's generated text.
*   **Rationale:** Provides transparency to the user about the information gathered and used by the LLM.
*   **Status:** Pending.

## Step 4: Error Handling and User Feedback

*   **File to Tweak:** `ChatWithLLM.tsx`.
*   **Action:** Implement robust error handling for the `/api/xeno-search` call.
    *   If the search fails, inform the user (e.g., "Xeno Search failed. Proceeding with standard response.").
    *   Provide clear loading states and feedback throughout the process.
*   **Rationale:** Ensures a good user experience even when issues occur.
*   **Status:** Pending.

## Step 5: Testing Frontend Integration

*   **Action:** Thoroughly test the entire flow:
    *   Toggling Xeno Search on/off.
    *   Sending queries with Xeno Search enabled.
    *   Verify loading states and display of search results.
    *   Verify that the LLM's response incorporates information from Xeno Search.
    *   Test error conditions (e.g., Python service down, Node.js endpoint error).
*   **Rationale:** Ensures the frontend integration is working correctly and robustly.
*   **Status:** Pending. 