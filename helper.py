from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy,time,os


class Helper(object):

    def check_file(self, filename, ext):
        file_ext = os.path.splitext(filename)[1]
        if file_ext.lower() not in ext:
            message = "Cannot Predict File "+file_ext+", Format File Not Allowed! Please Upload File .jpg"
            return message
        
    def tumor_predict(self, brain_image):
        # classes = []
        treshold = 0.5
        message_tumor = "Brain Has Tumor"
        message_no_tumor = "Brain Has No Tumor"
        model = load_model("models/vgg16_model_scen2confv1.h5")
        image = load_img("static/img_uploaded/"+brain_image, target_size = (224,224))
        x = img_to_array(image)
        x = numpy.expand_dims(x, axis = 0)
        
        # start_time = time.time()
        classification = model.predict(x)
        # pred_time = time.time() - start_time
        # pred_time = round(pred_time, 2)

        if classification <= treshold:
            # classes = [0]
            # classification = numpy.round(classification)
            result = message_no_tumor
        else:
            # classes = [1]
            # classification = numpy.round(classification)
            result = message_tumor
        
        # acc,pre,sen,f1score = self.perform_model(classes, classification)
        return result
    # , acc, pre, sen, f1score, pred_time
    
    # def perform_model(self, classes, classification):
    #     conf_mat = confusion_matrix(classes, classification, labels = [0,1])
    #     true_positive = conf_mat[0][0]
    #     false_positive = conf_mat[0][1]
    #     false_negative = conf_mat[1][0]
    #     true_negative = conf_mat[1][1]

    #     accuracy = round((true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative) * 100)
    #     if true_positive > true_negative:
    #         acc = accuracy
    #         pre = round(true_positive / (true_positive + false_positive) * 100)
    #         sen = round(true_positive / (true_positive + false_negative) * 100)
    #         # spe = round(true_negative / (true_negative + false_positive) * 100)
    #         f1score = round(2 * (pre * sen) / (pre + sen))
    #     else:
    #         acc = accuracy
    #         pre = round(true_negative / (true_negative + false_negative) * 100)
    #         sen = round(true_negative / (true_negative + false_positive) * 100)
    #         # spe = round(true_positive / (true_positive + false_negative) * 100, 1)
    #         f1score = round(2 * (pre * sen) / (pre + sen))
            
    #     return acc, pre, sen, f1score
    