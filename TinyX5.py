from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Lambda, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.engine.topology import Layer
import tensorflow as tf

class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh, 
                    grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, 
                    **kwargs):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        
        self.batch_size     = batch_size

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x
        #print ">>>>>>> y_true", y_true
        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
        #print ">>>>>>> y_pred", y_pred
        # initialize the masks
        object_mask     = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)        

        # compute grid factor and net factor
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        net_h       = tf.shape(input_image)[1]
        net_w       = tf.shape(input_image)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
        
        """
        Adjust prediction
        """
        pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
        pred_box_class = y_pred[..., 5:]                                                        # adjust class probabilities      

        """
        Adjust ground truth
        """
        true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
        true_box_wh    = y_true[..., 2:4] # t_wh
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)         
        #print ">>>> true_box_wh", true_box_wh
        """
        Compare each predicted box to all true boxes
        """        
        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0 

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious   = tf.reduce_max(iou_scores, axis=4)        
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """            
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor 
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half      

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
        
        count       = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
        class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50    = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
        recall75    = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)    
        avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
        avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
        avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3) 

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
                                       tf.ones_like(object_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])

        """
        Compare each true box to all anchor boxes
        """      
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
        conf_delta  = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + (1-object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                      self.class_scale

        loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
        loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
        loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
        loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))

#        loss = loss_xy + loss_wh + loss_conf + loss_class
        loss = (loss_xy + loss_wh + loss_conf + loss_class) / self.batch_size
        
        print "=============================================================================================================================================================="
        
        loss = tf.Print(loss, [grid_h, count, avg_obj, avg_noobj, avg_cat, avg_iou], message=' INFO: grid_h, count, avg_obj, avg_noobj, avg_cat, avg_iou\t', summarize=1000)
        loss = tf.Print(loss, [recall50, recall75], message=" RECALL: recall-50, recall-75\t", summarize=1000)
        loss = tf.Print(loss, [tf.reduce_sum(loss_xy), tf.reduce_sum(loss_wh), tf.reduce_sum(loss_conf), tf.reduce_sum(loss_class)], message=' LOSS: xy, wh, conf, class\t', summarize=1000)

        return loss*self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def _conv(inp, conv):
    x = inp
    # print ">> layer", conv['layer_idx'], x
    # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, 
    #       dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
    #       kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    if conv['init']:
        x = Conv2D(filters=conv['filter'],
                kernel_size=conv['kernel'],
                strides=conv['stride'],
                padding='same',
                name='conv_'+str(conv['layer_idx']),
                kernel_initializer='glorot_normal',
                use_bias=False if conv['bnorm'] else True)(x)
    else:
        x = Conv2D(filters=conv['filter'], 
                kernel_size=conv['kernel'], 
                strides=conv['stride'],
                padding='same', 
                name='conv_'+str(conv['layer_idx']), 
                use_bias=False if conv['bnorm'] else True)(x)

    if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['activation']=='leaky': x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    #print ">> layer", conv['layer_idx'], x
    return x 


def _maxpool(x, maxpool):
    return MaxPooling2D(pool_size=(maxpool['size'],maxpool['size']), 
                        strides=(maxpool['stride'], maxpool['stride']),
                        padding='same',
                        name='maxpool_'+str(maxpool['layer_idx']))(x)
     

def _upsample(x, upsample):
    return UpSampling2D(2, name='upsample_'+str(upsample['layer_idx']))(x)


def create_TinyX5_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, 
    batch_size, 
    warmup_batches,
    ignore_thresh,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale,
    init=False,
):
    input_image = Input(shape=(None, None, 3)) # net_h, net_w, 3   (min, 512, max)
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo_1 = Input(shape=(None, None, 3, 4+1+nb_class)) # len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(shape=(None, None, 3, 4+1+nb_class)) # len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
    # true_yolo_3 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class

    ## TinyX5 backbone
    x0  = _conv(input_image, {'layer_idx':  0, 'bnorm': True, 'filter':   16, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x1  = _maxpool(x0,       {'layer_idx':  1,                              'size':   2, 'stride': 2})
    x2  = _conv(x1,          {'layer_idx':  2, 'bnorm': True, 'filter':   32, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x3  = _maxpool(x2,       {'layer_idx':  3,                              'size':   2, 'stride': 2})
    x4  = _conv(x3,          {'layer_idx':  4, 'bnorm': True, 'filter':   64, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x5  = _maxpool(x4,       {'layer_idx':  5,                              'size':   2, 'stride': 2})
    x6  = _conv(x5,          {'layer_idx':  6, 'bnorm': True, 'filter':  128, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x7  = _maxpool(x6,       {'layer_idx':  7,                              'size':   2, 'stride': 2})
    x8  = _conv(x7,          {'layer_idx':  8, 'bnorm': True, 'filter':  256, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x9  = _maxpool(x8,       {'layer_idx':  9,                              'size':   2, 'stride': 2})
    x10 = _conv(x9,          {'layer_idx': 10, 'bnorm': True, 'filter':  512, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x11 = _maxpool(x10,      {'layer_idx': 11,                              'size':   2, 'stride': 1})
    x12 = _conv(x11,         {'layer_idx': 12, 'bnorm': True, 'filter': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x13 = _conv(x12,         {'layer_idx': 13, 'bnorm': True, 'filter':  256, 'kernel': 1, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x14 = _conv(x13,         {'layer_idx': 14, 'bnorm': True, 'filter':  512, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})

    ## yolo-layer-1 : layer 15 ==> 16
    pred_yolo_1 = _conv(x14, {'layer_idx': 15, 'bnorm':False, 'filter': 3*(4+1+nb_class), 'kernel': 1, 'stride': 1, 'pad': 1, 'activation': 'linear', 'init': init}) 
    loss_yolo_1 = YoloLayer(anchors[6:],
                            [1*num for num in max_grid],     ### ? not the feature size but the origin size, why?
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])
    print ">>>>>>>>>> loss_yolo_1", loss_yolo_1

    ## layer 17 ==> 21
    x17 = x13
    x18 = _conv(x17,         {'layer_idx': 18, 'bnorm': True, 'filter':  128, 'kernel': 1, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})
    x19 = _upsample(x18,     {'layer_idx': 19})
    x20 = concatenate([x19, x8])
    x21 = _conv(x20,         {'layer_idx': 21, 'bnorm': True, 'filter':  256, 'kernel': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky', 'init': init})


    ## yolo-layer-2 : layer 22 ==> 23
    pred_yolo_2 = _conv(x21, {'layer_idx': 22, 'bnorm':False, 'filter': 3*(4+1+nb_class), 'kernel': 1, 'stride': 1, 'pad': 1, 'activation': 'linear', 'init': init}) 
    loss_yolo_2 = YoloLayer(anchors[:6],
                            [2*num for num in max_grid],     ### ? not the feature size but the origin size, why?
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[1],     
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])
    print ">>>>>>>>>> loss_yolo_2", loss_yolo_2

    ## keras.Model(input, output): 
    ## -- train_model is to set train routine of specific network, so the output should be the loss for back-prop calculation
    ## -- infer_model only for forward calculation and focus on result, so the output should the prediction
    train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2], [loss_yolo_1, loss_yolo_2])
    infer_model = Model(input_image, [pred_yolo_1, pred_yolo_2])

    return [train_model, infer_model]


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))
