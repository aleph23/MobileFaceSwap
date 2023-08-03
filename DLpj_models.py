from facenet_pytorch import InceptionResnetV1
from face_detector import YoloDetector
import cv2
import torch
import numpy as np
from utils.align_face import align_img
import torch.nn as nn
import torchvision.transforms as T
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
model = YoloDetector(target_size = 720, device = "cuda:0",min_face = 20)


# def detection(img):
#   bboxes, points = model.predict(img)
#   # crop and align image
#   faces = model.align(img, points[0])

#   # Reshape tensor for resnet module
#   faces = torch.tensor(faces)
#   faces = faces.permute(0, 3, 1, 2)
#   faces = faces.float()
#   #bboxes = [float(num) for num in bboxes]
#   return faces, bboxes, points[0]

def detection(img, show=False, save_path=None, img_num=None):
  bboxes, points = model.predict(img)
  # crop and align image
  faces = model.align(img, points[0])
  # show pictures
  if show == True:
    for face in faces:
      cv2_imshow(face)
  # save pictures
  if save_path != None:
    for i in range(len(faces)):
      img = faces[i]
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = Image.fromarray(img)
      img.save(f'{save_path}/crop{img_num}_{i}.png')
  # Reshape tensor for resnet module
  faces = torch.tensor(faces)
  if faces.dim() == 4:
    faces = faces.permute(0, 3, 1, 2)
    faces = faces.float()
  #bboxes = [float(num) for num in bboxes]
    return faces, bboxes, points
  else:
    return None, None, None

def get_embeddings(faces):
  faces = faces.to(device)
  return resnet(faces).detach().cpu()

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
  point_list = [
      point for point, face_id in zip(points, face_ids) if face_id == 'unknown'
  ]
  point_list = np.array(point_list)
  return point_list


def process_image(img, target, recog_thr=0.4, version=1, view_sim=False): 
  _, _, points = detection(img)
  face_ids, _ = preprocess(img, target, recog_thr, version)
  return k(img, points[0], face_ids)




############################################ 수정

#defining the network

class Tuning(nn.Module):

    def __init__(self):
        super(Tuning,self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)

    )

    def forward(self,x):
        x = resnet(x)
        x = self.classifier(x)
        return x


path = f"{os.getcwd()}/model_v1.pt"

model_dl = torch.load(path, map_location=device)
model_dl.eval()

def process_image_dl(img): 
  data_transform = T.Compose([
      T.ToTensor(),
      T.Resize(244),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  imgs, _, points = detection(img)
  crop_img_list = []
  for i in imgs:
    i = T.ToPILImage()(i)
    crop_img_list.append(data_transform(i))

  crop_img_tensor = torch.stack(crop_img_list, dim=0)
  crop_img_tensor = crop_img_tensor.to(device)
  output = model_dl(crop_img_tensor)
  max_row_index = torch.argmax(output[:, 0])

  face_ids = []

  for i in range(len(output)):
    if i != max_row_index:
      face_ids.append("unknown")
    if i == max_row_index:
      face_ids.append("charm_zu")

  return k(img, points[0], face_ids)