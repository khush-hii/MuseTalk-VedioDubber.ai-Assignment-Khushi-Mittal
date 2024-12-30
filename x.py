import argparse
import cv2
from gfpgan import GFPGANer
from codeformer.inference_codeformer import enhance_image

# Initialize GFPGAN and CodeFormer
gfpgan_model = GFPGANer(
    model_path="GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth",
    upscale=4, arch="clean", channel_multiplier=2
)

def apply_gfpgan(image):
    _, _, restored_img = gfpgan_model.enhance(image, has_aligned=False, only_center_face=False)
    return restored_img

def apply_codeformer(image):
    restored_img = enhance_image(
        image, weight=0.7, pretrained_path="CodeFormer/weights/CodeFormer.pth"
    )
    return restored_img

def calculate_resolution_ratio(original_frame, generated_subframe):
    orig_h, orig_w = original_frame.shape[:2]
    gen_h, gen_w = generated_subframe.shape[:2]
    ratio_h = orig_h / gen_h
    ratio_w = orig_w / gen_w
    return max(ratio_h, ratio_w)

def process_video(input_video, input_audio, output_video, superres):
    # Load video frames and process lipsynced region
    cap = cv2.VideoCapture(input_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = None
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Assume `generated_subframe` is obtained here (modify based on MuseTalk's processing)
        generated_subframe = frame[100:200, 100:200]  # Example cropped region
        resolution_ratio = calculate_resolution_ratio(frame, generated_subframe)

        if resolution_ratio > 1:
            if superres == "GFPGAN":
                enhanced_frame = apply_gfpgan(generated_subframe)
            elif superres == "CodeFormer":
                enhanced_frame = apply_codeformer(generated_subframe)
            else:
                enhanced_frame = generated_subframe
        else:
            enhanced_frame = generated_subframe

        # Replace subframe back into the original frame (example)
        frame[100:200, 100:200] = enhanced_frame

        # Write output video
        if out is None:
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        out.write(frame)

    cap.release()
    if out:
        out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--superres", choices=["GFPGAN", "CodeFormer"], required=True, help="Superresolution method")
    parser.add_argument("-iv", "--input_video", required=True, help="Input video path")
    parser.add_argument("-ia", "--input_audio", required=True, help="Input audio path")
    parser.add_argument("-o", "--output_video", required=True, help="Output video path")
    args = parser.parse_args()

    process_video(args.input_video, args.input_audio, args.output_video, args.superres)
