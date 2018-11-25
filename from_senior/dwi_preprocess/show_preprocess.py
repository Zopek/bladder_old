import numpy as np
import matplotlib.pyplot as plt
import plot_3D_image
import full_preprocess
import preprocess_util
if __name__ == '__main__':
    root_dir = '/DATA3_DB7/data/public/renji_data/bladder_cleaned_distinct_series/'
    series_dir_list = [
        # 'D0719976/dwi_ax_0',
        # 'D1624257/dwi_ax_0',
        # 'D2049831/dwi_ax_0',
        # 'D2291117/dwi_ax_0',
        # 'D0645094/dwi_ax_1',
        # 'W0312093/dwi_ax_0',
        # 'D2314620/dwi_ax_0',
        # 'D0667043/dwi_ax_0',
        # 'D1598531/dwi_ax_0',
        # 'D1647256/dwi_ax_0',
        # 'D0501566/dwi_ax_0',
        '/D0643108/dwi_ax_0',
    ]

    for series_dir in series_dir_list:
        whole_adc, whole_b0, whole_b1000, mask, dilated_mask \
            = full_preprocess.full_preprocess(root_dir + series_dir, False)
        masked_adc = whole_adc * mask
        masked_b0 = whole_b0 * mask
        dilated_masked_adc = whole_adc * dilated_mask
        dilated_masked_b0 = whole_b0 * dilated_mask
        mask = mask[preprocess_util.find_bounding_box(mask)]

        def transpose_and_flip(a):
            return np.flip(np.flip(a.transpose((2, 1, 0)), 1), 2)
        fig = plt.figure()
        plane = plot_3D_image.Multi3DArrayPlane(fig, 3, 3)
        plane.add(transpose_and_flip(masked_adc))
        plane.add(transpose_and_flip(dilated_masked_adc))
        plane.add(transpose_and_flip(whole_adc))

        plane.add(transpose_and_flip(masked_b0))
        plane.add(transpose_and_flip(dilated_masked_b0))
        plane.add(transpose_and_flip(whole_b0))

        plane.add(transpose_and_flip(mask))
        plane.add(transpose_and_flip(dilated_mask))
        plane.add(transpose_and_flip(whole_b1000))
        plane.ready()
        fig.show()
    plt.show()
