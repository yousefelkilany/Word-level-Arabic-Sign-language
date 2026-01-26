import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.metrics import classification_report, confusion_matrix

from core.constants import DEVICE, FEAT_DIM, FEAT_NUM, SEQ_LEN
from core.utils import AR_WORDS, EN_WORDS
from data.data_preparation import DataAugmentor
from data.shared_elements import get_visual_controls
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


def render_inspector_view(rnd_key, dataloader, model=None):
    st.subheader("Inference with model")

    total_samples = len(dataloader.dataset)
    (
        idx,
        draw_lines,
        draw_points,
        separate_view,
        active_slices,
    ) = get_visual_controls(total_samples, rnd_key)

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
                kps, {part_name: sl}, draw_lines, draw_points, title=part_name.upper()
            )
            with cols[i]:
                st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(
            plot_3d_animation(kps, active_slices, draw_lines, draw_points),
            width="stretch",
        )


def render_augmentation_view(rnd_key, dataloader):
    st.subheader("🛠️ Augmentation Lab")
    total_samples = len(dataloader.dataset)
    (
        idx,
        draw_lines,
        draw_points,
        separate_view,
        active_slices,
    ) = get_visual_controls(total_samples, rnd_key)

    kps, lbl = dataloader.dataset[idx]

    config = {
        "hflip": False,
        "affine": False,
        "rotate": 0.0,
        "scale": 1.0,
        "tx": 0.0,
        "ty": 0.0,
    }
    (col_settings,) = st.columns([1])
    with col_settings:
        st.write("Transformation Settings")
        c1, c2 = st.columns(2)

        config["hflip"] = c1.checkbox("Horizontal Flip", value=False)
        config["affine"] = c2.checkbox("Affine Transform", value=False)

    if config["affine"]:
        ac1, ac2, ac3, ac4 = st.columns(4)
        config["rotate"] = ac1.slider("Rotate (°)", -45.0, 45.0, 0.0, 1.0)
        config["scale"] = ac2.slider("Scale", 0.5, 1.5, 1.0, 0.05)
        config["tx"] = ac3.slider("Shift X", -0.5, 0.5, 0.0, 0.05)
        config["ty"] = ac4.slider("Shift Y", -0.5, 0.5, 0.0, 0.05)

    original_kps, lbl = dataloader.dataset[idx]

    augmentor = DataAugmentor(
        p_flip=1 if config["hflip"] else 0,
        p_affine=1 if config["affine"] else 0,
        rotate_range=(-config["rotate"], config["rotate"]),
        scale_range=(-config["scale"], config["scale"]),
        shiftx_range=(-config["tx"], config["tx"]),
        shifty_range=(-config["ty"], config["ty"]),
    )
    aug_kps = original_kps.copy()
    aug_kps = augmentor(aug_kps)

    col_orig, col_aug = st.columns(2)

    with col_orig:
        original_kps = original_kps.reshape(SEQ_LEN, FEAT_NUM, FEAT_DIM)

        fig_orig = plot_3d_animation(
            original_kps, active_slices, draw_lines, draw_points, "Original"
        )
        st.plotly_chart(fig_orig, width="stretch")

    with col_aug:
        aug_kps = aug_kps.reshape(SEQ_LEN, FEAT_NUM, FEAT_DIM)

        fig_aug = plot_3d_animation(
            aug_kps, active_slices, draw_lines, draw_points, "Augmented"
        )
        st.plotly_chart(fig_aug, width="stretch")
