# 1. equivariant backbone compiling
cd models/vgtk && python setup.py install && cd ..
pip install -e .

# 2. pointnet++_lib compiling
cd pointnet_lib/ && python setup.py install && cd ../../

# 3. compile chamfer distance
cd utils/extensions/chamfer_dist
python setup.py install --user
