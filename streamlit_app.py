import random
import time
from typing import Dict, Generator, Iterable, List, Tuple

import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="📊 Sorting Algorithm Visualizer",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Utility Functions
# -----------------------------
ArrayState = Tuple[List[int], Dict[int, str]]


def render_chart(
    placeholder: st.delta_generator.DeltaGenerator,
    data: Iterable[int],
    highlights: Dict[int, str],
    title: str,
) -> None:
    colors = [
        highlights.get(idx, "#4682B4")  # steelblue default
        for idx in range(len(data))
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(range(len(data))),
                y=list(data),
                marker_color=colors,
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Index",
        yaxis_title="Value",
        title=title,
        template="plotly_white",
        height=400,
    )

    placeholder.plotly_chart(fig, config={"responsive": True})


def generate_random_list(size: int) -> List[int]:
    # Allow repeated numbers for a more realistic dataset but keep the range reasonable.
    return [random.randint(10, 99) for _ in range(size)]


# -----------------------------
# Sorting Algorithm Generators
# Each generator yields (state, highlights)
# -----------------------------
def bubble_sort(arr: List[int]) -> Generator[ArrayState, None, None]:
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            highlights = {j: "#FF4136", j + 1: "#FF4136"}  # red for comparisons
            yield arr.copy(), highlights
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                highlights = {j: "#2ECC40", j + 1: "#2ECC40"}  # green for swap
                yield arr.copy(), highlights
        if not swapped:
            break
    yield arr.copy(), {}


def insertion_sort(arr: List[int]) -> Generator[ArrayState, None, None]:
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        highlights = {i: "#FF851B"}
        yield arr.copy(), highlights
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
            highlights = {j + 1: "#FF4136", j + 2: "#FF4136"}
            yield arr.copy(), highlights
        arr[j + 1] = key
        highlights = {j + 1: "#2ECC40"}
        yield arr.copy(), highlights
    yield arr.copy(), {}


def selection_sort(arr: List[int]) -> Generator[ArrayState, None, None]:
    n = len(arr)
    for i in range(n):
        min_idx = i
        highlights = {i: "#0074D9"}  # blue for current position
        yield arr.copy(), highlights
        for j in range(i + 1, n):
            highlights = {min_idx: "#FF851B", j: "#FF4136"}
            yield arr.copy(), highlights
            if arr[j] < arr[min_idx]:
                min_idx = j
                highlights = {min_idx: "#2ECC40", i: "#FF851B"}
                yield arr.copy(), highlights
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            highlights = {i: "#2ECC40", min_idx: "#2ECC40"}
            yield arr.copy(), highlights
    yield arr.copy(), {}


def merge_sort(arr: List[int]) -> Generator[ArrayState, None, None]:
    def merge_sort_recursive(start: int, end: int) -> Generator[ArrayState, None, None]:
        if end - start <= 1:
            return
        mid = (start + end) // 2
        yield from merge_sort_recursive(start, mid)
        yield from merge_sort_recursive(mid, end)

        left = arr[start:mid]
        right = arr[mid:end]
        i = j = 0
        for k in range(start, end):
            if j >= len(right) or (i < len(left) and left[i] <= right[j]):
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            highlights = {idx: "#2ECC40" for idx in range(start, k + 1)}
            yield arr.copy(), highlights

    yield from merge_sort_recursive(0, len(arr))
    yield arr.copy(), {}


def quick_sort(arr: List[int]) -> Generator[ArrayState, None, None]:
    def partition(low: int, high: int) -> Generator[ArrayState, None, int]:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            highlights = {j: "#FF4136", high: "#FF851B"}
            yield arr.copy(), highlights
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                highlights = {i: "#2ECC40", j: "#2ECC40", high: "#FF851B"}
                yield arr.copy(), highlights
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        highlights = {i + 1: "#2ECC40", high: "#2ECC40"}
        yield arr.copy(), highlights
        return i + 1

    def quick_sort_recursive(low: int, high: int) -> Generator[ArrayState, None, None]:
        if low < high:
            pivot_index = yield from partition(low, high)
            yield from quick_sort_recursive(low, pivot_index - 1)
            yield from quick_sort_recursive(pivot_index + 1, high)

    yield from quick_sort_recursive(0, len(arr) - 1)
    yield arr.copy(), {}


def heap_sort(arr: List[int]) -> Generator[ArrayState, None, None]:
    def heapify(n: int, i: int) -> Generator[ArrayState, None, None]:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            highlights = {i: "#2ECC40", largest: "#2ECC40"}
            yield arr.copy(), highlights
            yield from heapify(n, largest)
        else:
            highlights = {i: "#FF851B"}
            yield arr.copy(), highlights

    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(n, i)

    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        highlights = {0: "#2ECC40", i: "#2ECC40"}
        yield arr.copy(), highlights
        yield from heapify(i, 0)

    yield arr.copy(), {}


ALGORITHMS = {
    "Bubble Sort": bubble_sort,
    "Insertion Sort": insertion_sort,
    "Selection Sort": selection_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort,
    "Heap Sort": heap_sort,
}


# -----------------------------
# Session State Initialization
# -----------------------------
DEFAULT_SIZE = 15
if "data" not in st.session_state:
    st.session_state["data"] = generate_random_list(DEFAULT_SIZE)
    st.session_state["list_size"] = DEFAULT_SIZE
    st.session_state["sorted"] = False


# -----------------------------
# Sidebar Controls
# -----------------------------
st.title("📊 Sorting Algorithm Visualizer")
st.write(
    "Interactively explore how different sorting algorithms work by watching "
    "their step-by-step execution on a random list."
)

algo_name = st.selectbox("Choose an algorithm", list(ALGORITHMS.keys()))
size = st.slider("Number of elements", min_value=5, max_value=50, value=st.session_state["list_size"])
delay_ms = st.slider("Visualization delay (ms)", min_value=0, max_value=1000, value=200, step=50)

col_generate, col_sort = st.columns(2)


with col_generate:
    if st.button("Generate New List"):
        st.session_state["data"] = generate_random_list(size)
        st.session_state["list_size"] = size
        st.session_state["sorted"] = False
        st.rerun()

with col_sort:
    start_sort = st.button("Start Sorting", type="primary")

if size != st.session_state["list_size"]:
    st.info("Generate a new list to apply the updated list size.", icon="ℹ️")


# -----------------------------
# Visualization Placeholder
# -----------------------------
chart_placeholder = st.empty()


def run_sort_visualization() -> None:
    data = st.session_state["data"].copy()
    algorithm = ALGORITHMS[algo_name]
    delay_seconds = delay_ms / 1000.0

    for state, highlights in algorithm(data):
        render_chart(
            chart_placeholder,
            state,
            highlights,
            title=f"{algo_name} Progress",
        )
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    st.session_state["data"] = data
    st.session_state["sorted"] = True


if start_sort:
    try:
        run_sort_visualization()
    except Exception as exc:
        st.error(f"Something went wrong during sorting: {exc}")
else:
    render_chart(
        chart_placeholder,
        st.session_state["data"],
        {},
        title=f"{algo_name} Ready",
    )


# -----------------------------
# Final Output
# -----------------------------
st.subheader("Current List")
st.write(st.session_state["data"])

if st.session_state.get("sorted"):
    st.success("Sorting complete! The list is now sorted.")


