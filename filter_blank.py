import util as u

masks_tr = u.masks(train=True)
ids_tr = u.img_ids(train=True)

f = open('blanks_tr.txt', 'w')
for id, mask in zip(ids_tr, masks_tr):
	if mask.max() == 0:
		f.write(id + '\n')
f.close()

masks_te = u.masks(train=False)
ids_te = u.img_ids(train=False)

f = open('blanks_te.txt', 'w')
for id, mask in zip(ids_te, masks_te):
	if mask.max() == 0:
		f.write(id + '\n')
f.close()
