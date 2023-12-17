import argparse
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import random
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image 

from deepface import DeepFace


#Import YOLO 
#from ultralytics import YOLO


device = torch.device('cpu')
# Since I do not have CUDA GPU 
#device = torch.device('cuda:0')

def filter_image_paths(image_paths):
    new_image_paths = []
    new_images = []
    for p in image_paths:
        img = np.array(Image.open(p).convert('RGB'))
        if img is None:
            continue
        new_image_paths.append(p)
        new_images.append(img)
    return new_image_paths, new_images


def get_features(model, preprocess, d, batch_size=32, is_valid_file=None, sort=False): 
    #print(args.model)
    dataset = datasets.ImageFolder(d, transform=preprocess, is_valid_file=is_valid_file) 
    if args.model == "vggface": 
        paths = [p for (p, _) in dataset.samples]
        target_size = (224, 224)
        all_features = [] 
        embeddings_users = []
        for path in paths: 
            try: 
                #  embedding_objs = DeepFace.represent(img_path = path, enforce_detection=False)
                #  embeddings = embedding_objs[0]["embedding"]
                #  embeddings_users.append(embeddings)
                # print(len(embeddings))
            
                faces = DeepFace.extract_faces(img_path=path, target_size=target_size, enforce_detection=False) 
                if faces: 
                    face = faces[0]["face"]
                    face = np.expand_dims(face, axis=0)
                    features = model.predict(face)[0, :]
                    all_features.append(features)
                else: 
                    print(f"No face detected in {path}")
            except Exception as e: 
                print(f"Error processing {path}: {str(e)}") 
        
        #return np.concatenate(all_features) 
        return np.vstack(all_features) 
        #return np.vstack(embeddings_users)

    if args.model == "facenet":  
        paths = [p for (p, _) in dataset.samples]
        target_size = (160, 160)
        all_features = [] 
        for path in paths: 
            try: 
                faces = DeepFace.extract_faces(img_path=path, target_size=target_size, enforce_detection=False) 
                if faces: 
                    face = faces[0]["face"]
                    face = np.expand_dims(face, axis=0)
                    features = model.predict(face)[0, :]
                    all_features.append(features)
                else: 
                    print(f"No face detected in {path}")
            except Exception as e: 
                print(f"Error processing {path}: {str(e)}")         
        return np.vstack(all_features) 

    elif args.model == "arcface": 
        paths = [p for (p, _) in dataset.samples]
        target_size = (112, 112) 
        all_features = [] 
        for path in paths: 
            try: 
                faces = DeepFace.extract_faces(img_path=path, target_size=target_size, enforce_detection=False) 
                if faces: 
                    face = faces[0]["face"]
                    face = np.expand_dims(face, axis=0)
                    features = model.predict(face)[0, :]
                    all_features.append(features)
                else: 
                    print(f"No face detected in {path}")
            except Exception as e: 
                print(f"Error processing {path}: {str(e)}") 
        return np.vstack(all_features) 


    elif args.model == "sface": 
        paths = [p for (p, _) in dataset.samples]
        target_size = (224, 224) 
        all_features = [] 
        for path in paths: 
            try: 
                faces = DeepFace.extract_faces(img_path=path, target_size=target_size, enforce_detection=False) 
                if faces: 
                    face = faces[0]["face"]
                    face = np.expand_dims(face, axis=0)
                    features = model.predict(face)[0, :]
                    all_features.append(features)
                else: 
                    print(f"No face detected in {path}")
            except Exception as e: 
                print(f"Error processing {path}: {str(e)}") 
        return np.vstack(all_features) 


    elif args.model == "fawkesv10":
        paths = [p for (p, _) in dataset.samples]
        paths, images = filter_image_paths(paths)
        faces = model.faces(paths, images, None, verbose=0, eval_local=True, no_align=True)
        faces = faces.cropped_faces
        features = model.predict(faces).numpy()
        return features 

    else:
        if sort:
            loader_users = DataLoader(dataset, num_workers=1, batch_size=32, shuffle=False)
        else:
            loader_users = DataLoader(dataset, num_workers=4, batch_size=32)
        
        with torch.no_grad():
            all_classes = [] 
            embeddings_users = []
            for data_user, class_user in iter(loader_users):
                embeddings = model(data_user.to(device)).detach().cpu().numpy()
                embeddings_users.append(embeddings)
                all_classes.extend(class_user.cpu().numpy())
        return np.concatenate(embeddings_users)

# def get_class(data_dir):
#     folders_arr = data_dir.split('/')
#     for i in range(len(folders_arr)-1):
#         if folders_arr[i+1] == 'face':
#             class_name = folders_arr[i]
#             return class_name
#     return None



def get_class(data_dir):
    folders_arr = data_dir.split('/')
    
    # Check if 'face' is present in the path
    if 'face' in folders_arr:
        # If 'face' is present, find the class name before it
        for i in range(len(folders_arr)-1):
            if folders_arr[i+1] == 'face':
                class_name = folders_arr[i]
                return class_name
    else:
        # If 'face' is not present, consider the last folder as the class name
        if folders_arr:
            return folders_arr[-1]

    return None 

def get_facescrub_features(model, preprocess, dataset_path):
    # get features for all facescrub users
    data_dirs = sorted(glob.glob(os.path.join(dataset_path, "*")))

    classes_train = []
    features_train = []

    classes_test = []
    features_test = []

    for index, data_dir in enumerate(data_dirs): 
        cls = get_class(data_dir) 

        #print("data_dir: ", data_dir) 
        print("Index: " , index)
        print("Class: ", cls) 

        f = get_features(model, preprocess, data_dir)

        #print("f", f)

        test_len = int(0.3 * len(f))
        test_idx = random.sample(range(len(f)), test_len)

        f_test = f[test_idx]
        f_train = np.delete(f, test_idx, axis=0)
        features_train.append(f_train)
        classes_train.extend([cls] * len(f_train))
        features_test.append(f_test)
        classes_test.extend([cls] * len(f_test))

    classes_train = np.asarray(classes_train)
    features_train = np.concatenate(features_train, axis=0)

    classes_test = np.asarray(classes_test)
    features_test = np.concatenate(features_test, axis=0)

    return features_train, features_test, classes_train, classes_test

def main():

    if args.model == "magface": 

        # Add the parent directory to sys.path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
        sys.path.append(parent_directory) 

        from MagFace.inference.network_inf import builder_inf
        
        #from inference.network_inf import builder_inf

        #Adding cpu mode to args 
        args.cpu_mode = True 
        model = builder_inf(args) 
        model = model.to("cpu") 
        #Since I do not have Cuda GPU 
        #model = model.cuda()
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((112, 112), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ]) 
        
    elif args.model == "clip": 
        
        import clip
        m, preprocess = clip.load("ViT-B/32", device=device)

        import PIL
        preprocess = transforms.Compose([transforms.Resize(size=(224, 224)),
                                         *preprocess.transforms])

        model = lambda x: m.encode_image(x) 

    
    elif args.model == "vggface":
        model = DeepFace.build_model("VGG-Face")
        # preprocess = None
        preprocess = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor()]) 
                                        #*preprocess.transforms])

    elif args.model == "facenet":
        model = DeepFace.build_model("Facenet") 
        preprocess = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor()]) 
    
    elif args.model == "arcface":
        model = DeepFace.build_model("ArcFace")
        preprocess = None 
        # preprocess = transforms.Compose([transforms.Resize(size=(112, 112)), transforms.ToTensor()]) 

    elif args.model == "sface": 
        model = DeepFace.build_model("SFace")  
        preprocess = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])  


    elif args.model == "fawkesv10":
        #from fawkes.utils import init_gpu, load_extractor, load_victim_model, Faces 

        # Add the parent directory to sys.path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
        sys.path.append(parent_directory)

        # Now you can import the module 
        # For fawkes v1.0 extractor 
        from fawkes1.fawkes.utils import init_gpu, load_extractor, load_victim_model, Faces 


        sess = init_gpu("0")
        model = load_extractor("extractor_0")
        model.faces = Faces
        preprocess = None

    else: 
        print(f"Unsupported model: {args.model}") 

    
    random.seed(10)
    print("Extracting features...", flush=True)
    X_train_all, X_test_all, Y_train_all, Y_test_all = get_facescrub_features(model, preprocess, dataset_path=args.facescrub_dir)


    print(len(X_train_all), len(X_test_all), flush=True)

    val_people = args.names_list
    base_dir = args.attack_dir

    for name in val_people:
        directory = f"{base_dir}/{name}" 
        print(directory)

        filter_cloak = lambda x: args.protected_file_match in x
        filter_uncloak = lambda x: args.unprotected_file_match in x 
        
        f_cloaked = get_features(model, preprocess, directory, is_valid_file=filter_cloak)
        f_uncloaked = get_features(model, preprocess, directory, is_valid_file=filter_uncloak)
        print(name, len(f_cloaked), len(f_uncloaked), flush=True) 

        #assert len(f_cloaked) == len(f_uncloaked)


        random.seed(10)
        test_frac = 0.3
        test_idx = random.sample(range(len(f_cloaked)), int(test_frac * len(f_cloaked)))
        f_train_cloaked = np.delete(f_cloaked, test_idx, axis=0)
        f_test_cloaked = f_cloaked[test_idx]
        
        f_train_uncloaked = np.delete(f_uncloaked, test_idx, axis=0)
        f_test_uncloaked = f_uncloaked[test_idx]

        if args.classifier == "linear":
            clf1 = LogisticRegression(random_state=0, n_jobs=-1, warm_start=False)
            clf1 = make_pipeline(StandardScaler(), clf1)
        else:
            clf1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

        idx_train = np.asarray([y != name for y in Y_train_all])
        idx_test = np.asarray([y != name for y in Y_test_all])
        print(np.sum(idx_train), np.sum(idx_test))

        # with cloaking
        X_train = np.concatenate((X_train_all[idx_train], f_train_cloaked))
        Y_train = np.concatenate((Y_train_all[idx_train], [name] * len(f_train_cloaked))) 

        clf1 = clf1.fit(X_train, Y_train) 
        print("Test acc: {:.2f}".format(clf1.score(X_test_all[idx_test], Y_test_all[idx_test]))) 
        print("Train acc (user cloaked): {:.2f}".format(clf1.score(f_train_cloaked, [name] * len(f_train_cloaked))))
        print("Test acc (user cloaked): {:.2f}".format(clf1.score(f_test_uncloaked, [name] * len(f_test_uncloaked))))
        print("Protection rate: {:.2f}".format(1-clf1.score(f_test_uncloaked, [name] * len(f_test_uncloaked))))
        print()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        help='the feature extractor', default='magface')
    parser.add_argument('--classifier', type=str,
                        help='the classifier', default='NN')

    parser.add_argument('--names-list', nargs='+', default=[], help="names of attacking users")
    parser.add_argument('--facescrub-dir', help='path to unprotected facescrub directory', default="facescrub/download/")
    parser.add_argument('--attack-dir', help='path to protected facescrub directory', default="facescrub_attacked/download/")
    parser.add_argument('--unprotected-file-match', type=str,
                        help='pattern to match protected pictures', default='.jpg')
    parser.add_argument('--protected-file-match', type=str,
                        help='pattern to match protected pictures', default='high_cloaked.png')

    # for MagFace
    parser.add_argument('--arch', default='iresnet100', type=str,
                        help='backbone architechture')
    parser.add_argument('--embedding_size', default=512, type=int,
                        help='The embedding feature size')
    parser.add_argument('--resume', default="magface_epoch_00025.pth", 
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
