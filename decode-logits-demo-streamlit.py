from pathlib import Path

import numpy as np
import streamlit as st
import torch
from src import autoencoders

st.set_page_config(layout="wide")


# 画像生成関数
def generate_image(model, logit):
    model.eval()
    with torch.no_grad():
        output_images = model.forward_decoder(logit).numpy(force=True)[0][0]
    assert output_images.shape == (28, 28)

    return output_images


def set_preset(values):
    for i, value in enumerate(values):
        st.session_state[f"slider_{i}"] = float(value)


def main():
    st.title("Logits Decoder Demo")

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_paths = sorted(checkpoint_dir.glob("decoder*.pth"))
    if not checkpoint_paths:
        st.error(f"No decoder checkpoints found in {checkpoint_dir}")
        st.stop()

    checkpoint_name = st.selectbox(
        "モデルファイル",
        [path.name for path in checkpoint_paths],
    )
    chkpt = checkpoint_dir / checkpoint_name
    checkpoint_state = torch.load(
        chkpt,
        map_location=autoencoders.device,
        weights_only=True,
    )
    latent_dim = 10
    centroid_path = Path(f"{chkpt}_centroid.npy")
    centroids = np.load(centroid_path)
    if centroids.shape != (10, latent_dim):
        raise ValueError(
            f"Expected centroids with shape (10, {latent_dim}), "
            f"got {centroids.shape} from {centroid_path}"
        )

    model = autoencoders.DeepConvAutoencoder(central_dim=latent_dim)
    model.load_state_dict(checkpoint_state)
    model.to(autoencoders.device)

    st.caption(f"使用中のモデル: {chkpt.name}")

    left, right = st.columns([1, 1], gap="small")

    with left:
        logit_values = []
        _, preset_header = st.columns([5, 2], gap="small")
        with preset_header:
            st.markdown("Preset (class mean)")
        for i in range(10):
            col1, col2 = st.columns([5, 2], gap="small")
            with col1:
                val = st.slider(f"Logit {i}", min_value=-60.0, max_value=60.0, value=0.0, key=f"slider_{i}", label_visibility="collapsed")
                logit_values.append(val)
            with col2:
                st.button(str(i), key=f"preset_btn_{i}", on_click=set_preset, args=(centroids[i],))

    logit_tensor = torch.tensor([logit_values], dtype=torch.float32)
    img = generate_image(model, logit_tensor)

    with right:
        st.markdown(
            "<style>div[data-testid='stImage'] img { image-rendering: pixelated; }</style>",
            unsafe_allow_html=True,
        )
        st.image(img, caption="Decoded Image", use_container_width=True)


if __name__ == "__main__":
    main()