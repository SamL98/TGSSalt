import util as u
from iou import iou_metric

import numpy as np

model = u.load_unet()

all_ids = u.img_ids(train=True)

batch_size = 64
i = 0

id_batch = all_ids[i*batch_size : (i+1)*batch_size]
ious = dict()

while i*batch_size < len(all_ids)-1:
	batch, y = u.ips(for_ids=id_batch)
	if len(batch.shape) == 3:
		batch = np.expand_dims(batch, axis=3)

	p = model.predict(batch, batch_size=batch_size)[:,:,:,0]
	for img_id, true, pred in zip(id_batch, y, p):
		ious[img_id] = iou_metric(true, pred)

	i += 1
	id_batch = all_ids[i*batch_size : (i+1)*batch_size]

ious = sorted(ious, key=lambda kv: kv[1])
with open('ious.csv', 'w') as f:
	for iou_pair in ious:
		print('%s:%f' % (iou_pair[0], iou_pair[1]))
		f.write('%s:%f\n' % (iou_pair[0], iou_pair[1]))