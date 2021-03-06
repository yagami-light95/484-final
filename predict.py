# -*- coding: utf-8 -*-
from cnn_model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

save_path = os.path.join(Config.saver_folder, 'best_validation')


class img_predict:

    def __init__(self):
        self.model = CNN()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)

    @staticmethod
    def check_array(image):
        # Check the size of the image
        if image.shape != (60, 160):
            raise ValueError('Only 160*60 captcha-size is accepted.')

    def predict(self, captcha):
        image = Image.open(captcha).convert('L')
        image = np.array(image)
        self.check_array(image)
        image = image.flatten() / 255.0
        data = image.reshape([1, Config.width * Config.height])

        feed_dic = {self.model.input_x: data,
                    self.model.keep_prob: 1.0,
                    self.model.training: False}
        predict_label = self.session.run(self.model.predict_max_idx, feed_dict=feed_dic)
        return predict_label[0]


if __name__ == "__main__":
    img = 'test/4085.jpg'
    p = img_predict()
    result = p.predict(img)
    print'The img we want to predict: %s\n' %img
    print 'Predict result: %s' % (''.join([str(x) for x in result]))
