from facenet_pytorch import InceptionResnetV1
from face_detector import YoloDetector
import cv2
import torch
import numpy as np
from utils.align_face import align_img



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
model = YoloDetector(target_size = 720, device = "cuda:0",min_face = 20)


def detection(img):
  bboxes, points = model.predict(img)
  # crop and align image
  faces = model.align(img, points[0])

  # Reshape tensor for resnet module
  faces = torch.tensor(faces)
  faces = faces.permute(0, 3, 1, 2)
  faces = faces.float()
  #bboxes = [float(num) for num in bboxes]
  return faces, bboxes, points[0]


def get_embeddings(faces):
    faces = faces.to(device)
    unknown_embeddings = resnet(faces).detach().cpu()
    return unknown_embeddings

def recognition(face_db, unknown_embeddings, recog_thr) : 
    face_ids = []
    probs = []
    for i in range(len(unknown_embeddings)):
        result_dict = {}
        eb = unknown_embeddings[i]
        for name in face_db.keys():
            knownfeature_list = face_db[name]
            prob_list = []
            for knownfeature in knownfeature_list:
                prob = (eb - knownfeature).norm().item()
                prob_list.append(prob)
                if prob < recog_thr:
                    # 기준 넘으면 바로 break해서 같은 인물 계속 안 체크하도록
                    break
            result_dict[name] = min(prob_list)
        results = sorted(result_dict.items(), key=lambda d:d[1])
        result_name, result_probs = results[0][0], results[0][1]
        if result_probs < recog_thr: 
            face_ids.append(result_name)
        else : 
            face_ids.append('unknown')
        probs.append(result_probs)
    return face_ids, probs

def recognition_v2(face_db, unknown_embeddings, recog_thr) : 
    face_ids = []
    similarities = []
    for i in range(len(unknown_embeddings)):
        result_dict = {}
        eb = unknown_embeddings[i]
        for name in face_db.keys():
            knownfeature_list = face_db[name]
            similarity_list = []
            for knownfeature in knownfeature_list:
                similarity =  torch.nn.functional.cosine_similarity(eb, knownfeature, dim=0)
                similarity_list.append(similarity)
                if similarity > recog_thr:
                    # 기준 넘으면 바로 break해서 같은 인물 계속 안 체크하도록
                    break
            result_dict[name] = max(similarity_list)
        results = sorted(result_dict.items(), key=lambda d:d[1], reverse=True)
        result_name, result_similarity = results[0][0], results[0][1]
        if result_similarity > recog_thr: 
            face_ids.append(result_name)
        else : 
            face_ids.append('unknown')
        similarities.append(result_similarity)
    return face_ids, similarities

def recognition_v3(face_db, unknown_embeddings, recog_thr) : 
    face_ids = []
    probs = []
    for i in range(len(unknown_embeddings)):
        result_dict = {}
        eb = unknown_embeddings[i]
        for name in face_db.keys():
            knownfeature_list = face_db[name]
            prob_list = []
            for knownfeature in knownfeature_list:
                prob = (eb - knownfeature).norm().item()
                prob_list.append(prob)
            result_dict[name] = np.mean(prob_list)
        results = sorted(result_dict.items(), key=lambda d:d[1])
        result_name, result_probs = results[0][0], results[0][1]
        if result_probs < recog_thr: 
            face_ids.append(result_name)
        else : 
            face_ids.append('unknown')
        probs.append(result_probs)
    return face_ids, probs


def preprocess(img, target, recog_thr, version) :
  faces, bboxes, _ = detection(img)

  unknown_embeddings = get_embeddings(faces)
  if version == 1:
    face_ids, probs = recognition(target, unknown_embeddings, recog_thr)
  if version == 2:
    face_ids, probs = recognition_v2(target, unknown_embeddings, recog_thr)
  if version == 3:
    face_ids, probs = recognition_v3(target, unknown_embeddings, recog_thr)

  return face_ids, probs


def k(img, points, face_ids):
  point_list = []

  for (point, face_id) in zip(points, face_ids):
    if face_id == 'unknown':
        point_list.append(point)

  point_list = np.array(point_list)  
  return point_list


def process_image(img, target, recog_thr=0.42, version=3, view_sim=False): 
    _, bboxes, points = detection(img)
    face_ids, _ = preprocess(img, target, recog_thr, version)
    result = k(img, points, face_ids)

    return result


