import numpy as np
import json

def cosine_similarity(v1,v2):
    return (np.dot(v1,v2.T))/(np.linalg.norm(v1)*np.linalg.norm(v2))

def cosine_similarity_Matrix(v1,v2):
    norm1 = np.linalg.norm(v1, axis = 1).shape
    norm2 = np.linalg.norm(v2.T, axis = 0).shape

    norm1 = np.array([norm1])
    norm2 = np.array([norm2])
    return np.argmax((np.dot(v1,v2.T))/np.dot(norm1.T, norm2), axis = 1)

if __name__ == "__main__":
    #For Question 6
    with open('cnn_dataset.json') as data_file:
        data = json.load(data_file)
    pixel_mj1 = np.array([data['pixel_rep']['mj1']])
    pixel_mj2 = np.array([data['pixel_rep']['mj2']])
    pixel_cat = np.array([data['pixel_rep']['cat']])
    vgg_mj1 = np.array([data['vgg_rep']['mj1']])
    vgg_mj2 = np.array([data['vgg_rep']['mj2']])
    vgg_cat = np.array([data['vgg_rep']['cat']])
    cos_simi_pixel_mj1_mj2 = cosine_similarity(pixel_mj1, pixel_mj2)
    cos_simi_pixel_mj2_cat = cosine_similarity(pixel_mj2, pixel_cat)
    cos_simi_pixel_mj1_cat = cosine_similarity(pixel_mj1, pixel_cat)
    cos_simi_vgg_mj1_mj2 = cosine_similarity(vgg_mj1, vgg_mj2)
    cos_simi_vgg_mj2_cat = cosine_similarity(vgg_mj2, vgg_cat)
    cos_simi_vgg_mj1_cat = cosine_similarity(vgg_mj1, vgg_cat)
    print "The Cosine SImilarity between mj1 and mj2 in pixel representation is:\n " + str(cos_simi_pixel_mj1_mj2)
    print "The Cosine SImilarity between mj2 and cat in pixel representation is:\n " + str(cos_simi_pixel_mj2_cat)
    print "The Cosine SImilarity between mj1 and cat in pixel representation is:\n " + str(cos_simi_pixel_mj1_cat)
    print "The Cosine SImilarity between mj1 and mj2 in vgg representation is:\n " + str(cos_simi_vgg_mj1_mj2)
    print "The Cosine SImilarity between mj2 and cat in vgg representation is:\n " + str(cos_simi_vgg_mj2_cat)
    print "The Cosine SImilarity between mj1 and cat in vgg representation is:\n " + str(cos_simi_vgg_mj1_cat)
    print '\n'

    #For Question 8
    vgg_data = np.load('vgg_rep.npy', mmap_mode = 'r')
    pixel_data = np.load('pixel_rep.npy', mmap_mode = 'r')
    with open('dataset.json') as data_file:
        data_list = json.load(data_file)
    train_list = data_list['train']
    test_list = data_list['test']
    image_list = data_list['images']
    caption_list = data_list['captions']
    vgg_train_data = np.array([vgg_data[i, :] for i,j in enumerate(image_list) if j in train_list])
    vgg_test_data = np.array([vgg_data[i, :] for i,j in enumerate(image_list) if j in test_list])

    pixel_train_data = np.array([pixel_data[i, :] for i,j in enumerate(image_list) if j in train_list])
    pixel_test_data = np.array([pixel_data[i, :] for i,j in enumerate(image_list) if j in test_list])
    
    result = cosine_similarity_Matrix(vgg_test_data, vgg_train_data)

    vggFile = open("vgg.txt", 'w')
    for i, j in zip(result, test_list):
        str = caption_list[train_list[i]] + '\n'
        print caption_list[train_list[i]]
        vggFile.write(str)
    vggFile.close()
    print '\n'

    result = cosine_similarity_Matrix(pixel_test_data, pixel_train_data)

    pixelFile = open("pixel.txt", 'w')
    for i, j in zip(result, test_list):
        str = caption_list[train_list[i]] + '\n'
        print caption_list[train_list[i]]
        pixelFile.write(str)
    pixelFile.close()