from collections import deque

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from em_classification.calc_utils import calc_median, enrich_data

class IVT:
    def __init__(self, gaze_data, stim_data, EPS=0.85):
        stim_pos = stim_data[:, -2:]
        stim_timestamps = stim_data[:, 0]

        rad_data = enrich_data(gaze_data, stim_data)
        self.data = rad_data[['Timestamp', 'GazePointXLeft', 'GazePointYLeft', 'rp', 'diff_rad', 'drv', 'diff_x', 'diff_y']].copy()
        rad_data.to_csv('data_log.csv', sep='\t')
        self.timestamps = np.array(stim_timestamps)
        self.VT = self.get_adaptive_vt()
        self.fixations = self._calc_fix_bounds()
        self.EPS = EPS

    def get_adaptive_vt(self):
        return np.percentile(self.data['drv'], 85)

    def get_fix_medians(self, stim_fix, hor_field='GazePointXLeft', ver_field='GazePointYLeft'):
        fix_data = self._get_fix_data_slice(stim_fix)
        fix_medians = []
        for fix in stim_fix:
            beg, end = fix
            slice_data = fix_data[(fix_data.Timestamp >= beg) & (fix_data.Timestamp <= end)]
            hor_slice = slice_data[hor_field]
            ver_slice = slice_data[ver_field]
            hor_spread = hor_slice.median()
            ver_spread = ver_slice.median()
            fix_medians.append([hor_spread, ver_spread])

        return fix_medians

    def cluster_to_merge(self, fix_medians):
        if len(fix_medians) == 0:
            return []
        db = DBSCAN(eps=self.EPS, min_samples=2).fit(fix_medians)
        return db.labels_

    def get_calib_fixations(self):
        all_fix = list(self.fixations)
        calib_fix = []
        all_merged_fix = []
        next_stim_fix = self._extract_stim_fix(all_fix, self.timestamps[0], self.timestamps[1])
        T = len(self.timestamps)
        for i in range(1, T):
            print(i)
            stim_fix = next_stim_fix

            merges_count = 0
            changed = True
            while changed:
                changed = False
                # print('BEFORE:', stim_fix)
                prev_fix_num = len(stim_fix)
                stim_fix = self._merge_close_fix(stim_fix)
                merges_count += 1
                changed = len(stim_fix) != prev_fix_num
                if merges_count == 3:
                    exit()
                # print('AFTER:', stim_fix)

            if i < T - 1:
                beg_next_ts = self.timestamps[i]
                end_next_ts = self.timestamps[i + 1]
                next_stim_fix = self._extract_stim_fix(all_fix, beg_next_ts, end_next_ts)
                # copy the list of fixations on the next calibration target, so
                # in case the merging will end up with one collapsed fixation
                # we can revert back
                if len(stim_fix) > 0:
                    temp = next_stim_fix.copy()
                    # preprend last fixation of current target to the next target:
                    temp.insert(0, stim_fix[-1])
                    temp = self._merge_close_fix(temp)
                    # if fixations didn't collapse into one
                    if len(temp) > 1:
                        next_stim_fix = temp
                        # append last merged fixation back to current target:
                        stim_fix[-1] = next_stim_fix.pop(0)

            # find the fixation 'closest' to calibration target
            # discard all small fixations:
            stim_fix = self._discard_small_fixations(stim_fix)
            fix = self._find_closest_fix(stim_fix)
            calib_fix.append(fix)
            all_merged_fix.extend(stim_fix)

        # calib_fix = self._discard_small_fixations(calib_fix)
        # all_merged_fix = self._discard_small_fixations(all_merged_fix)

        return calib_fix, all_merged_fix

    def _discard_small_fixations(self, fixations):
        filtered_fixations = []
        for fix in fixations:
            beg, end = fix
            # discard fixation if it's less than 50ms (6 samples)
            if end - beg < 2 * (8 + 8 + 9):
                continue
            filtered_fixations.append(fix)
        return filtered_fixations

    def _DEBUG(self, stim_fix, mesg):
        print(mesg)
        fix_data = self._get_fix_data_slice(stim_fix)
        for fix in stim_fix:
            beg, end = fix
            slice_data = fix_data[
                (fix_data.Timestamp >= beg) & (fix_data.Timestamp <= end)
            ]
            hor_slice = slice_data['GazePointXLeft']
            ver_slice = slice_data['GazePointYLeft']
            print(fix, hor_slice.median(), ver_slice.median())

    def _extract_stim_fix(self, all_fix, stim_beg_ts, stim_end_ts):
        N = len(all_fix)

        beg_ind = 0
        while beg_ind < N and all_fix[beg_ind][0] < stim_beg_ts:
            beg_ind += 1

        end_ind = beg_ind
        while end_ind < N and all_fix[end_ind][0] < stim_end_ts:
            end_ind += 1

        return list(all_fix[beg_ind: end_ind])

    def _find_closest_fix(self, stim_fix):
        best_val = np.inf
        best_fix = (np.nan, np.nan)

        for fix in stim_fix:
            hor_med = calc_median(self.data, *fix, 'diff_x', take_abs=True)
            ver_med = calc_median(self.data, *fix, 'diff_y', take_abs=True)
            val = max(hor_med, ver_med) / (10 * (fix[1] - fix[0]))
            print(fix, val)
            if val < best_val:
                best_fix = fix
                best_val = val

        return best_fix

    # def _find_closest_fix_(self, stim_fix):
    #     best_med = np.inf
    #     best_ind = -1

    #     for ind, fix in enumerate(stim_fix):
    #         med = calc_median(self.data, *fix, 'diff_rad')
    #         if med < best_med:
    #             best_ind = ind
    #             best_med = med

    #     if best_ind == -1:
    #         return (np.nan, np.nan)

    #     for ind, fix in enumerate(stim_fix):
    #         best_len = stim_fix[best_ind][1] - stim_fix[best_ind][0]
    #         fix_len = fix[1] - fix[0]
    #         if best_len < fix_len:
    #             hor_med = calc_median(self.data, *fix, 'diff_x', take_abs=True)
    #             ver_med = calc_median(self.data, *fix, 'diff_y', take_abs=True)
    #             if (np.abs(hor_med) + np.abs(ver_med) < self.HOR_DT + self.VER_DT):
    #                 best_ind = ind

    #     return stim_fix[best_ind]

    def _merge_close_fix(self, stim_fix, hor_field='GazePointXLeft', ver_field='GazePointYLeft'):
        stim_fix = list(stim_fix)

        stim_fix_medians = self.get_fix_medians(stim_fix)
        cluster_labels = self.cluster_to_merge(stim_fix_medians)
        print(cluster_labels)

        i = 0
        label_ind = 0

        while i < len(stim_fix):
            curr_label = cluster_labels[label_ind]
            if curr_label == -1:
                label_ind += 1
                i += 1
                continue

            j = label_ind
            while j < len(cluster_labels) and curr_label == cluster_labels[j]:
                j += 1

            merge_len = (j - label_ind)
            label_ind += merge_len

            if merge_len > 1:
                first = i
                last = i + merge_len - 1
                stim_fix = self._merge_all_from_first_to_last(stim_fix, first, last)

            i += 1

        return stim_fix

    def _merge_all_from_first_to_last(self, stim_fix, first, last):
        print('MERGE, from:', stim_fix[first], 'to:', stim_fix[last])
        beg = stim_fix[first][0]
        end = stim_fix[last][1]
        # delete previous and current fixations that are merged
        del stim_fix[first: last + 1]
        # add the merged fixation
        new_fix = (beg, end)
        stim_fix.insert(first, new_fix)
        return stim_fix

    def _calc_fix_bounds(self):
        self.data['mask'] = np.array(self.data['drv'] < self.VT, dtype=np.int)
        fix_bounds = []

        prev_row = self.data.iloc[0]
        fix_beg = prev_row['Timestamp'] if prev_row['mask'] == 1 else None

        for _, row in self.data.iloc[1:].iterrows():
            if prev_row['mask'] == 1 and row['mask'] == 0:
                fix_end = prev_row['Timestamp']
                fix_bounds.append((fix_beg, fix_end))
            if prev_row['mask'] == 0 and row['mask'] == 1:
                fix_beg = row['Timestamp']
            prev_row = row
        
        if prev_row['mask'] == 1:
            fix_end = prev_row['Timestamp']
            fix_bounds.append((fix_beg, fix_end))

        return fix_bounds

    def _get_fix_data_slice(self, fix_bounds):
        all_fix = deque(fix_bounds)
        fix_data = pd.DataFrame(columns=[*self.data.columns])

        for fix in all_fix:
            beg, end = fix
            slice_data = self.data[
                (self.data.Timestamp >= beg) & (self.data.Timestamp <= end)
            ]
            fix_data = pd.concat([fix_data, slice_data], ignore_index=True)

        return fix_data
