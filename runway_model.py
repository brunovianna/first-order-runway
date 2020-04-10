import runway
import os
import yaml
import face_alignment

import numpy as np
from skimage.transform import resize

import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp


@runway.setup(options={'checkpoint': runway.file(extension='.tar')})
def setup(opts):
    #config_path = 'config/vox-256.yaml'
    config_path = 'config/fashion-256.yaml'
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.cuda()

    checkpoint_path = opts['checkpoint']
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    #fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    return generator, kp_detector


def crop_face(fa, pil_image, pad=150):
    w, h = pil_image.size
    landmarks = np.stack(fa.get_landmarks(np.array(pil_image))).squeeze()
    left, upper = landmarks.min(0)
    right, lower = landmarks.max(0)
    bbox = [
        max(0, int(left - pad)),
        max(0, int(upper - pad)),
        min(w, int(right + pad)),
        min(h, int(lower + pad))
    ]
    bbox_size = [
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
    ]
    return pil_image.crop(bbox), bbox, bbox_size


def convert_to_torch_tensor(pil_image):
    np_img = resize(np.array(pil_image), (256, 256))
    return torch.tensor(np_img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()


kp_driving_initial = None

@runway.command('generate', inputs={'driving_image': runway.image, 'source_image': runway.image}, outputs={'output': runway.image})
def generate(model, inputs):
    global kp_driving_initial
    #generator, kp_detector, fa = model
    generator, kp_detector = model
    source_image = inputs['source_image']
    #cropped_source_image, _, _ = crop_face(fa, source_image)
    #source = convert_to_torch_tensor(cropped_source_image)
    source = convert_to_torch_tensor(source_image)
    #driving = convert_to_torch_tensor(crop_face(fa, inputs['driving_image'])[0])
    driving = convert_to_torch_tensor(inputs['driving_image'])
    kp_driving = kp_detector(driving)
    kp_source = kp_detector(source)
    if kp_driving_initial is None:
        kp_driving_initial = kp_driving
    kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                            kp_driving_initial=kp_driving_initial, use_relative_movement=True,
                            use_relative_jacobian=True, adapt_movement_scale=True)
    out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
    out = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
    out = (out*255).astype(np.uint8)
    # out = Image.fromarray(out).resize(crop_size)
    # source_image.paste(out, crop_bbox)
    return out


if __name__ == '__main__':
    #runway.run(host='localhost', port=8888, debug=True, model_options={'checkpoint': './vox-cpk.pth.tar'})
    runway.run(host='localhost', port=8888, debug=True, model_options={'checkpoint': './fashion.pth.tar'})
