from collections import deque

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from em_classification.calc_utils import calc_median, enrich_data

class IVT:
    def __init__(self, gaze_data, stim_data, EPS=0.85, verbose=False):
        stim_pos = stim_data[:, -2:]
        stim_timestamps = stim_data[:, 0]

        self.data = gaze_data.copy()
        self.data = enrich_data(self.data, stim_data)
        mask = [
            'time', 'pos_x', 'pos_y', 'diff_x', 'diff_y',
            'pos_rad', 'diff_rad', 'rv'
        ]
        self.data = self.data[mask]

        self.EPS = EPS
        self.VT_PERCENT = 85
        self.VT = self.get_adaptive_vt()
        # fixations that are shorter than 50ms (6 samples) will be discarded
        self.SMALL_FIX_TH = 2 * (8 + 8 + 9)

        self.data.to_csv('data_log.csv', sep='\t')
        self.timestamps = np.array(stim_timestamps)
        self.fixations = self._calc_fix_bounds()

        self.verbose = verbose

    def log_to_console(self, *args, **kargs):
        if self.verbose:
            print(*args, **kargs)

    def get_adaptive_vt(self):
        return np.percentile(self.data['rv'], self.VT_PERCENT)

    def get_fix_medians(self, stim_fix, hor_label='pos_x', ver_label='pos_y'):
        fix_data = self._get_fix_data_slice(stim_fix)
        fix_medians = []
        for fix in stim_fix:
            beg, end = fix
            slice_data = fix_data[(fix_data.time >= beg) & (fix_data.time <= end)]
            hor_slice = slice_data[hor_label]
            ver_slice = slice_data[ver_label]
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
            self.log_to_console('Target #', i)
            stim_fix = next_stim_fix

            stim_fix = self._merge_close_fix(stim_fix)

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

        return calib_fix, all_merged_fix

    def _discard_small_fixations(self, fixations):
        filtered_fixations = []
        for fix in fixations:
            beg, end = fix
            # discard fixation if it's less than 50ms (6 samples)
            if end - beg < self.SMALL_FIX_TH:
                continue
            filtered_fixations.append(fix)
        return filtered_fixations

    def _DEBUG(self, stim_fix, mesg):
        print(mesg)
        fix_data = self._get_fix_data_slice(stim_fix)
        for fix in stim_fix:
            beg, end = fix
            slice_data = fix_data[
                (fix_data.time >= beg) & (fix_data.time <= end)
            ]
            hor_slice = slice_data['pos_x']
            ver_slice = slice_data['pos_y']
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
            self.log_to_console(fix, val)
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

    def _merge_close_fix(self, stim_fix):
        stim_fix = list(stim_fix)

        stim_fix_medians = self.get_fix_medians(stim_fix)
        cluster_labels = self.cluster_to_merge(stim_fix_medians)
        self.log_to_console(cluster_labels)

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
        self.log_to_console('MERGE, from:', stim_fix[first], 'to:', stim_fix[last])
        beg = stim_fix[first][0]
        end = stim_fix[last][1]
        # delete previous and current fixations that are merged
        del stim_fix[first: last + 1]
        # add the merged fixation
        new_fix = (beg, end)
        stim_fix.insert(first, new_fix)
        return stim_fix

    def _calc_fix_bounds(self):
        self.data['mask'] = np.array(self.data['rv'] < self.VT, dtype=np.int)
        fix_bounds = []

        prev_row = self.data.iloc[0]
        fix_beg = prev_row['time'] if prev_row['mask'] == 1 else None

        for _, row in self.data.iloc[1:].iterrows():
            if prev_row['mask'] == 1 and row['mask'] == 0:
                fix_end = prev_row['time']
                fix_bounds.append((fix_beg, fix_end))
            if prev_row['mask'] == 0 and row['mask'] == 1:
                fix_beg = row['time']
            prev_row = row
        
        if prev_row['mask'] == 1:
            fix_end = prev_row['time']
            fix_bounds.append((fix_beg, fix_end))

        return fix_bounds

    def _get_fix_data_slice(self, fix_bounds):
        all_fix = deque(fix_bounds)
        fix_data = pd.DataFrame(columns=[*self.data.columns])

        for fix in all_fix:
            beg, end = fix
            slice_data = self.data[
                (self.data.time >= beg) & (self.data.time <= end)
            ]
            fix_data = pd.concat([fix_data, slice_data], ignore_index=True)

        return fix_data
