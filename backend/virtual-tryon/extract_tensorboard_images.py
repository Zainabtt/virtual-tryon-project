import os
from tensorboard.backend.event_processing import event_accumulator # type: ignore
from PIL import Image
import io

event_file = r"C:\Users\IMOE001\VFR\cp-vton\tensorboard\tom_train_new\events.out.tfevents.1745572646.DESKTOP-RCMU0GD"

ea = event_accumulator.EventAccumulator(event_file, size_guidance={'images': 0})
ea.Reload()

image_tags = ea.Tags().get('images', [])
print("✅ Available image tags:", image_tags)

if image_tags:
    selected_tag = image_tags[0]  # أول تاج
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    for img_summary in ea.Images(selected_tag):
        step = img_summary.step
        img_str = img_summary.encoded_image_string
        image = Image.open(io.BytesIO(img_str))
        output_path = os.path.join(output_dir, f"{selected_tag.replace('/', '_')}_step_{step}.png")
        image.save(output_path)
        print(f"✅ Saved: {output_path}")
else:
    print("⚠️ No image tags found in the event file.")
