from model import *
from labelgen import *
from tensorflow.python import debug as tf_debug

saver = tf.train.Saver()
tf.train.export_meta_graph(filename=model_dir+".meta")

trainop = tf.train.AdamOptimizer(0.001).minimize(total_loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,model_dir)
#vggsaver.restore(sess,'./vgg16/vgg16')
# def ddd(da,t):
#     return t.dtype!=np.object and np.max(np.abs(t)).tolist()>1e4
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#sess.add_tensor_filter('mf',ddd)
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")
#saver.restore(sess,model_dir)

for i in range(epochs):
    print('----------epoch:'+str(i))
    for j,(im,positivemask,samplingmask,verticallabel) in enumerate(traindatagen()):
        _err,_ec,_ey,_log,_ = sess.run([total_loss,conf_loss,y_loss,logging,trainop],feed_dict={input_ph:im,positive_mask:positivemask,sampling_mask:samplingmask,verti_gt:verticallabel})
        writer.add_summary(_log)
        print(_err,_ec,_ey)
        if j%10==0:
            try:
                saver.save(sess, model_dir, write_meta_graph=False)
            except Exception:
                traceback.print_exc()


