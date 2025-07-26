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


def main():
    centroids = [
        [35.818363, -44.152626, 1.9347773, -17.854965, -19.310617,
         -7.363521, 4.8163834, -2.850933, -4.4936824, -5.942543],
        [-16.14973, 29.623085, -5.4714017, -8.9712305, 3.8225226,
         -8.781906, -3.3743305, 5.599071, 7.9834976, -16.154554],
        [0.3499025, 2.9045432, 44.329613, 8.47216, -8.481991,
         -23.094193, -23.522676, 10.196703, 7.4585257, -20.853275],
        [-23.678614, -3.1880505, 3.727031, 47.427414, -34.963394,
         8.03907, -37.630165, -13.9076, 6.657253, 13.705978],
        [-23.681948, -3.5699213, -10.549996, -36.07146, 38.091293,
         -11.647012, -11.0007925, -4.2511954, -7.5848026, 1.7114027],
        [-19.7545, -13.922253, -24.961702, 5.256699, -19.964169,
         38.002014, -6.1270313, -32.464333, 5.0571337, 0.28905332],
        [7.2296543, -17.900803, -17.725622, -15.078482, -4.304661,
         7.5350685, 41.437016, -37.56466, 1.8869267, -19.287302],
        [-2.4174616, 2.1017387, -6.6017637, -6.718111, -7.114219,
         -18.749353, -31.32972, 33.221577, -14.521173, 5.032805],
        [-4.376863, -7.182853, 3.9939585, 1.2086406, -15.63735,
         -4.556414, -9.653619, -20.005278, 34.449116, -2.3611429],
        [-18.859976, -18.08829, -19.800066, -0.23623233, 2.7391813,
         -9.9626, -31.745888, -2.051464, -3.4292893, 28.952946]]

    epochs = 200
    bs = 64
    latent_dim = 10
    chkpt = f'checkpoints/decoder_bs{bs}_ep{epochs}_latent{latent_dim}.pth'

    model = autoencoders.DeepConvAutoencoder(central_dim=latent_dim)
    model.load_state_dict(torch.load(chkpt, map_location=autoencoders.device, weights_only=True))
    model.to(autoencoders.device)

    st.title("Logits Decoder Demo")

    left, right = st.columns([1, 1], gap="small")

    with left:
        logit_values = []
        preset_index = st.session_state.get("preset_index", 0)
        if "use_preset" in st.session_state:
            sliders = centroids[preset_index]
        else:
            sliders = [st.session_state.get(f"slider_{i}", 0.0) for i in range(10)]
        for i in range(10):
            col1, col2 = st.columns([6, 1], gap="small")
            with col1:
                val = st.slider(f"Logit {i}", min_value=-60.0, max_value=60.0, value=float(sliders[i]), key=f"slider_{i}", label_visibility="collapsed")
            with col2:
                if st.button(f"Preset {i}", key=f"preset_btn_{i}"):
                    st.session_state["preset_index"] = i
                    st.session_state["use_preset"] = True
                    st.rerun()
            logit_values.append(val)
        if "use_preset" in st.session_state:
            del st.session_state["use_preset"]

    logit_tensor = torch.tensor([logit_values], dtype=torch.float32)
    img = generate_image(model, logit_tensor)

    with right:
        st.image(img, caption="Decoded Image", use_container_width=True)


if __name__ == "__main__":
    main()