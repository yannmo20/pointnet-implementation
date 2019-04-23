import json
import os

def main():
    path = '/media/moellya/yannick/labels'
    files = os.listdir(path)
    classes = set()
    for cnt, file in enumerate(files, 1):
        with open(os.path.join(path, file), 'r') as f:
            objects = json.load(f)['children'][0]['children'][0]['children']
        if cnt % 100 == 0:
            print(cnt)
        for obj in objects:
            classes.add(obj['identity'])
    print(classes)


if __name__ == '__main__':
    main()
