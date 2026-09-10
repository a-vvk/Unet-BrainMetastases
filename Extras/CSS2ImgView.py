import numpy as np
import matplotlib.pyplot as plt

data = np.load('segmentation5.npy')
print("Volume shape:", data.shape)

norm_data = data / np.max(data)

axial_index = data.shape[2] // 2     
coronal_index = data.shape[1] // 2    
sagittal_index = data.shape[0] // 2   

plt.figure()
plt.imshow(norm_data[:, :, axial_index], cmap='gray')
plt.title(f"Axial Slice {axial_index}")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(norm_data[:, coronal_index, :], cmap='gray')
plt.title(f"Coronal Slice {coronal_index}")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(norm_data[sagittal_index, :, :], cmap='gray')
plt.title(f"Sagittal Slice {sagittal_index}")
plt.axis('off')
plt.show()