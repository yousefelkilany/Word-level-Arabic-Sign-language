import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.metrics import classification_report, confusion_matrix

from core.constants import DEVICE
from core.mediapipe_utils import KP2SLICE
from core.utils import AR_WORDS, EN_WORDS
from modelling.dashboard.visualization import plot_3d_animation


def render_metrics_view(y_true, y_pred, num_words):
    acc = (y_true == y_pred).mean()
    st.metric("Overall Accuracy", f"{acc:.2%}")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    df_classes = df_report[df_report.index.str.isdigit()].copy()
    df_classes["Label_AR"] = [AR_WORDS[int(i)] for i in df_classes.index]
    df_classes["Label_EN"] = [EN_WORDS[int(i)] for i in df_classes.index]

    st.dataframe(
        df_classes.style.background_gradient(
            subset=["f1-score"], cmap="RdYlGn", vmin=0, vmax=1
        ),
        width="stretch",
    )

    cm = confusion_matrix(y_true, y_pred)
    labels_text = [f"{i}: {EN_WORDS[i]}" for i in range(num_words)]
    fig_cm = px.imshow(
        cm,
        x=labels_text,
        y=labels_text,
        color_continuous_scale="Blues",
        labels=dict(color="Count"),
    )
    fig_cm.update_layout(height=800)
    st.plotly_chart(fig_cm, width="stretch")


def render_error_view(y_true, y_pred, y_probs):
    st.subheader("Top Confused Pairs")
    errors_mask = y_true != y_pred
    if not np.any(errors_mask):
        st.success("No errors found!")
        return

    df_errors = pd.DataFrame(
        {
            "True_Word": [EN_WORDS[i] for i in y_true[errors_mask]],
            "Pred_Word": [EN_WORDS[i] for i in y_pred[errors_mask]],
        }
    )

    counts = (
        df_errors.groupby(["True_Word", "Pred_Word"]).size().reset_index(name="Count")  # ty:ignore[no-matching-overload]
    )
    counts = counts.sort_values("Count", ascending=False).head(20)

    fig = px.bar(
        counts,
        x="Count",
        y="True_Word",
        color="Pred_Word",
        orientation="h",
        title="Top 20 Misclassifications",
    )
    st.plotly_chart(fig, width="stretch")


def render_inspector_view(dataloader, model=None):
    st.subheader("Deep Dive")
    col_sel, col_checks = st.columns([1, 2])
    total_samples = len(dataloader.dataset)

    with col_sel:
        idx = st.number_input(
            f"Sample Index (0 - {total_samples - 1})",
            min_value=0,
            max_value=total_samples - 1,
            value=0,
            help=f"Select a sample from 0 to {total_samples - 1}",
        )

    with col_checks:
        st.write("Visual Controls:")
        c_m, c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1, 1])
        draw_edges = c_m.checkbox("Edges", True)
        separate_view = c_m.checkbox("Separate Parts", True)

        show_pose = c1.checkbox("Body", True)
        show_face = c2.checkbox("Face", True)
        show_rh = c3.checkbox("RH", False)
        show_lh = c4.checkbox("LH", False)

    active_slices = {}
    if show_pose:
        active_slices["pose"] = KP2SLICE["pose"]
    if show_face:
        active_slices["face"] = KP2SLICE["face"]
    if show_rh:
        active_slices["rh"] = KP2SLICE["rh"]
    if show_lh:
        active_slices["lh"] = KP2SLICE["lh"]

    kps, lbl = dataloader.dataset[idx]

    if model is not None:
        st.markdown("##### Model is not None")
        inp = torch.tensor(kps).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)
            pred_idx = int(torch.argmax(probs).item())

        true_en, pred_en = EN_WORDS[int(lbl)], EN_WORDS[pred_idx]
        conf_val = probs[0][pred_idx].item()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("True Label")
            st.markdown(f"### {true_en}")

        with c2:
            st.caption("Predicted Label")
            if true_en == pred_en:
                st.markdown(f"### :green[{pred_en}] ✅")
            else:
                st.markdown(f"### :red[{pred_en}] ❌")

        with c3:
            st.caption("Confidence")

            if conf_val > 0.8:
                bar_color = "#21c354"
            elif conf_val > 0.5:
                bar_color = "#ffa421"
            else:
                bar_color = "#ff4b4b"

            st.markdown(
                f"<h3 style='margin:0; padding:0;'>{conf_val:.2%}</h3>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="
                    background-color: rgba(128, 128, 128, 0.2); 
                    border-radius: 5px; 
                    height: 8px; 
                    width: 100%; 
                    margin-top: 8px;">
                    <div style="
                        background-color: {bar_color}; 
                        width: {conf_val * 100}%; 
                        height: 100%; 
                        border-radius: 5px; 
                        transition: width 0.5s ease-in-out;">
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if separate_view and len(active_slices) > 0:
        cols = st.columns(len(active_slices))
        for i, (part_name, sl) in enumerate(active_slices.items()):
            fig = plot_3d_animation(
                kps, {part_name: sl}, draw_edges, title=part_name.upper()
            )
            with cols[i]:
                st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(
            plot_3d_animation(kps, active_slices, draw_edges),
            width="stretch",
        )
