import os


# ========================================================================
# Train
route_tr = '/home/torres/DUTS-TR'
tr_rw = os.path.join(route_tr, 'DUTS-TR-Image')
tr_gt = os.path.join(route_tr, 'DUTS-TR-Mask')

dest_tr = open('DUTS-train_ids.csv', 'w')
dest_tr.write('ID\n')
files = os.listdir(tr_rw)
for file_id in files:
    file_id = file_id.replace('.jpg','')
    dest_tr.write('{}\n'.format(file_id))
    print('Succesfully written information for file {}\n'.format(file_id))
dest_tr.close()
# ========================================================================
# Test
route_te = '/home/torres/DUTS-TE'
te_rw = os.path.join(route_te, 'DUTS-TE-Image')
te_gt = 'DUTS-TE-Mask'
dest_te = open('DUTS-test_ids.csv', 'w')
dest_te.write('ID\n')
files = os.listdir(te_rw)
for file_id in files:
    file_id = file_id.replace('.jpg','')
    dest_te.write('{}\n'.format(file_id))
    print('Succesfully written information for file {}\n'.format(file_id))
dest_te.close()

