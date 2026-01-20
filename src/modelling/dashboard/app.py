import os

import streamlit as st

from core.constants import DEVICE, TRAIN_CHECKPOINTS_DIR
from modelling.dashboard.loader import (
    get_checkpoints_num_words,
    get_split_dataloader,
    load_cached_checkpoints,
    load_cached_model,
    run_inference,
)
from modelling.dashboard.views import (
    render_error_view,
    render_inspector_view,
    render_metrics_view,
)

st.set_page_config(layout="wide", page_title="KArSL Analytics")


def main():
    st.sidebar.title("KArSL Dashboard")

    ckpt_files = load_cached_checkpoints(TRAIN_CHECKPOINTS_DIR)
    if ckpt_files is None:
        st.sidebar.error(f"Checkpoints dir not found: {TRAIN_CHECKPOINTS_DIR}")

    selected_ckpt, num_words, model = None, 502, None
    if len(ckpt_files) == 0:
        st.sidebar.info("No checkpoints found.")
    else:
        selected_ckpt = st.sidebar.selectbox("Select Checkpoint", ckpt_files)
        if not selected_ckpt:
            st.sidebar.info("Select a checkpoint")
        else:
            selected_ckpt = os.path.join(TRAIN_CHECKPOINTS_DIR, selected_ckpt)
            num_words = get_checkpoints_num_words(selected_ckpt)
            model = load_cached_model(selected_ckpt, num_words)

    split_select = st.sidebar.radio("Split", ["train", "val", "test"], index=0)
    dataloader = get_split_dataloader(num_words, split_select)

    if "current_view" not in st.session_state:
        st.session_state.current_view = None

    if "results" not in st.session_state:
        st.session_state.results = None

    if selected_ckpt and st.sidebar.button("Run Evaluation", type="primary"):
        st.session_state.results = run_inference(
            model, dataloader, DEVICE, selected_ckpt, split_select
        )

    default_tabs_views = ["Global Metrics", "Error Analysis", "Sample Inspector"]
    tabs_views = default_tabs_views[-1:]

    y_true, y_pred, y_probs = (None,) * 3
    if st.session_state.results:
        y_true, y_pred, y_probs = st.session_state.results
        tabs_views = default_tabs_views[:]
    elif selected_ckpt:
        st.info("Click 'Run Evaluation' in the sidebar to start.")

    tabs = st.tabs(tabs_views)
    inspector_tab = tabs[-1]
    if y_true is not None and y_pred is not None and y_probs is not None:
        metrics_tab, errors_tab, inspector_tab = tabs
        with metrics_tab:
            render_metrics_view(y_true, y_pred, num_words)
        with errors_tab:
            render_error_view(y_true, y_pred, y_probs)

    with inspector_tab:
        render_inspector_view(dataloader, model)

    if "current_eye" not in st.session_state:
        st.session_state.current_eye = None

    # view = st.radio(
    #     "View",
    #     tabs_views,
    #     horizontal=True,
    #     label_visibility="collapsed",
    # )
    # st.divider()

    # render_view = st.session_state.current_view or view
    # if render_view == "Metrics":
    #     st.session_state.current_view = render_view
    #     render_metrics_view(y_true, y_pred, num_words)
    # elif render_view == "Errors":
    #     st.session_state.current_view = render_view
    #     render_error_view(y_true, y_pred, y_probs)
    # elif render_view == "Inspector":
    #     st.session_state.current_view = render_view
    #     render_inspector_view(dataloader, model)


if __name__ == "__main__":
    main()
