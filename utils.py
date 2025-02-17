import numpy as np
import cv2

def draw_mask(mask, random_color=False, fontscale=0.3, draw_mask=True, mark_instances=True, mark_instances_pos='topleft', draw_instance_count=True):
    assert mark_instances in [True, False, 'marker', 'number'], f"mark_instances should be either True, False, 'marker', or 'number', but got {mark_instances}"
    if mark_instances:
        assert mark_instances_pos in ['topleft', 'center'], f"mark_instances_pos should be either 'topleft' or 'center', but got {mark_instances_pos}"

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.1])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = np.zeros((h, w, 3))

    # Draw mask
    if draw_mask:
        mask_image[mask > 0] = color[:3]

    # Draw numbers on each instance
    if not mark_instances:
        pass
    elif mark_instances == 'marker':  # Draw markers
        for i in np.unique(mask)[1:]:  # Skip background
            _loc = np.where(mask == i)

            if mark_instances_pos == 'topleft':
                # Find the top-left corner of the bounding box
                _loc = zip(_loc[1], _loc[0])
                _loc = sorted(_loc, key=lambda x: x[0])
                _orig = _loc[0]
            elif mark_instances_pos == 'center':
                # Find the center of the bounding box
                _orig = (int(np.mean(_loc[1])), int(np.mean(_loc[0])))

            # Draw marker
            cv2.drawMarker(mask_image, (_orig[0], _orig[1]), (255, 255, 255), markerType=cv2.MARKER_DIAMOND, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
    else:  # Write numbers
        for i in np.unique(mask)[1:]:  # Skip background
            _loc = np.where(mask == i)

            if mark_instances_pos == 'topleft':
                # Find the top-left corner of the bounding box
                _loc = zip(_loc[1], _loc[0])
                _loc = sorted(_loc, key=lambda x: x[0])
                _orig = _loc[0]
            elif mark_instances_pos == 'center':
                # Find the center of the bounding box
                _orig = (int(np.mean(_loc[1])), int(np.mean(_loc[0])))

            # Draw number
            (_w, _h), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, fontscale, 1)
            cv2.rectangle(mask_image, (_orig[0] - 1, _orig[1] + 1), (_orig[0] + _w + 1, _orig[1] - _h - 1), color, -1)
            cv2.rectangle(mask_image, (_orig[0] - 1, _orig[1] + 1), (_orig[0] + _w + 1, _orig[1] - _h - 1), (255, 255, 255), 1)
            cv2.putText(mask_image, str(i), _orig, cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 255, 255), 1, cv2.LINE_AA) 

    # Draw instance count
    if draw_instance_count:
        n_instances = max(np.unique(mask))
        n_instances_vis = len(np.unique(mask)) - 1
        # bottom-left corner
        _orig = (0, h)
        (_w, _h), _ = cv2.getTextSize(f'{n_instances_vis}/{n_instances}', cv2.FONT_HERSHEY_SIMPLEX, fontscale * 1.5, 1)
        cv2.rectangle(mask_image, (_orig[0] - 1, _orig[1] + 1), (_orig[0] + _w + 1, _orig[1] - _h - 1), color, -1)
        cv2.rectangle(mask_image, (_orig[0] - 1, _orig[1] + 1), (_orig[0] + _w + 1, _orig[1] - _h - 1), (255, 255, 255), 1)
        cv2.putText(mask_image, f'{n_instances_vis}/{n_instances}', _orig, cv2.FONT_HERSHEY_SIMPLEX, fontscale * 1.5, (255, 255, 255), 1, cv2.LINE_AA)

    return mask_image

def show_mask(mask, axes, alpha, *args, **kwargs):
    mask_image = draw_mask(mask, *args, **kwargs)
    axes.imshow(mask_image, alpha=alpha)