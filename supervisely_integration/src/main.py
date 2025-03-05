import supervisely as sly
from dotenv import load_dotenv
from supervisely_integration.src.birefnet import BiRefNet


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv("supervisely.env")

model = BiRefNet(
    use_gui=True,
    use_serving_gui_template=True,
    sliding_window_mode="none",
)
model.serve()
