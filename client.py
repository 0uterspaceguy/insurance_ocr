import os
import requests
from tqdm import tqdm

for image_name in tqdm(os.listdir('examples')):
    image_path = os.path.join('examples',image_name)
    data = open(image_path,'rb').read()
    r = requests.post("http://127.0.0.1:5000/recognize/",data=data)
    print(r.text)

