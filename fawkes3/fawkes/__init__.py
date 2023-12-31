# -*- coding: utf-8 -*-
# @Date    : 2020-07-01
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/


__version__ = '0.3.1'

from .detect_faces import create_mtcnn, run_detect_face
from .differentiator import FawkesMaskGeneration
from .protection import main, Fawkes
from .utils import load_extractor, init_gpu, select_target_label, dump_image, reverse_process_cloaked, Faces, get_file, \
    filter_image_paths


# from detect_faces import create_mtcnn, run_detect_face
# from differentiator import FawkesMaskGeneration
# from protection import main, Fawkes
# from utils import load_extractor, init_gpu, select_target_label, dump_image, reverse_process_cloaked, Faces, get_file, \
#     filter_image_paths 

__all__ = (
    '__version__', 'create_mtcnn', 'run_detect_face',
    'FawkesMaskGeneration', 'load_extractor',
    'init_gpu',
    'select_target_label', 'dump_image', 'reverse_process_cloaked',
    'Faces', 'get_file', 'filter_image_paths', 'main', 'Fawkes'
)
