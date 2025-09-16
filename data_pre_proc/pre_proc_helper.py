#%%
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
import matplotlib.pyplot as plt
import re

from pathlib import Path
from pprint import pprint
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, sosfiltfilt, medfilt

import maestro_file
import json
hv.extension('bokeh')


class single_trial:
    POS_NORMALIZER = 40
    VEL_NORMALIZER = 10.8826

    def __init__(self, file_path):
        self.file_path = file_path
        for key, value in self.get_file_data(file_path).items():
            setattr(self, key, value)

    def __repr__(self):
        return json.dumps(self.data_dict, default=str, indent=4)
    
    def __str__(self):
        return f"Single Trial: {self.trial_name} {self.filename}"

    @property
    def data_dict(self):
        # iterate over all attributes and return as dictionary
        atrrs = [atrr for atrr in dir(self) if not atrr.startswith("__") and not callable(atrr)]
        [
            atrrs.remove(func) for func in [
                'data_dict', 'get_file_data', 
                'extract_trail_info_from_trial_name', 'plot_behavior',
                'get_saccades', 'get_first_relevant_saccade',
                'from_dict', 'POS_NORMALIZER', 'VEL_NORMALIZER',
                '_single_trial__first_relevant_saccade'
        ]]
        data_dict = {
            attr: getattr(self, attr) 
            for attr in atrrs
        }
        full_path = self.file_path.parts
        data_dir_idx = full_path.index('data')
        data_dict['file_path'] = str(Path(*full_path[data_dir_idx:]))
        return data_dict
        
    
    @property
    def first_relevant_saccade(self):
        try:
            return self.__first_relevant_saccade
        except AttributeError:
            self.get_first_relevant_saccade()
            return self.__first_relevant_saccade

    def get_file_data(self, file_path: Path) -> dict:
        try:
            with open(file_path, 'rb') as f:
                data_file = maestro_file.DataFile.load(f.read(), file_path.name)
        except FileNotFoundError:
            other_path = Path(*list(file_path.parts[:-5] + file_path.parts[-4:]))
            with open(other_path, 'rb') as f:
                data_file = maestro_file.DataFile.load(f.read(), file_path.name)
            self.file_path = other_path
            

        # Extract trial information using regex
        trial_info = self.extract_trail_info_from_trial_name(data_file.trial.name)
        
        # Handle case where trial name doesn't match expected patterns
        if trial_info is None:
            Warning(f"Trial name '{data_file.trial.name}' doesn't match expected patterns")
            trial_type = None
            direction = None
            ssd_number = None
            dir_value = None
        else:
            trial_type = trial_info.get("type")
            direction = trial_info.get("direction")
            ssd_number = trial_info.get("ssd_number")  # Will be None for GO trials
            # Convert direction to dir value: R=0, L=180
            dir_value = 0 if direction == "R" else 180 if direction == "L" else None
        
        # Todo: crop out first segment interval from behaivoral data, spikes
        hVel = np.array(data_file.ai_data[2]) / self.VEL_NORMALIZER
        vVel = np.array(data_file.ai_data[3]) / self.VEL_NORMALIZER
        
        segs_times = np.array([0] + [seg.dur for seg in data_file.trial.segments]).cumsum()

        trail_row = {
            'filename': data_file.file_name, # e.g., 'fi211109a.2040'
            'trail_session': data_file.file_name.split('.')[0], # e.g., 'fi211109a'
            'trail_number': data_file.file_name.split('.')[1], # e.g., '2040'
            'trial_name': data_file.trial.name, # e.g., GO_R, STOP_L_SSD2
            'set': data_file.trial.set_name,    # CSST, 8dir_saccade, ...
            'type': trial_type,                    # GO, STOP, or CONT
            'direction': direction,                # R or L
            'ssd_number': ssd_number,             # 1-4 for STOP/CONT, None for GO
            'ssd_len': data_file.trial.segments[2].dur,  # length of SSD segment
            'segs_durations': np.array([seg.dur for seg in data_file.trial.segments]), # durations of all segments
            'segs_times': segs_times, # cumulative times of segment boundaries
            'go_cue': segs_times[2], # time of go cue in ms
            'stop_cue': segs_times[3] if trial_type in ['STOP', 'CONT'] else None, # time of stop (or continue) cue in ms
            'dir': dir_value,                     # 0 for R, 180 for L
            'hPos': np.array(data_file.ai_data[0]) / self.POS_NORMALIZER, # in degrees
            'vPos': np.array(data_file.ai_data[1]) / self.POS_NORMALIZER, # in degrees
            'hVel': hVel, # in degrees/sec
            'vVel': vVel, # in degrees/sec
            'speed': np.sqrt(vVel**2 + hVel**2) , # in degrees/sec
            'trial_length': len(data_file.ai_data[0]), # in ms
            'blinks': data_file.blinks, # list of (start, end) times in ms. Should be None
            'trial_failed': not bool(data_file.header.flags & maestro_file.FLAG_REWARD_GIVEN), # True if trial was failed
            'neural_data': data_file.sorted_spikes   # dict of spike times keyed by cell_id
        }    
        return trail_row

    def extract_trail_info_from_trial_name(self, text):
        """
        Extract information from three types of strings:
        1. GO_{R/L} -> returns direction
        2. STOP_{R/L}_SSD{1/2/3/4} -> returns direction and number
        3. CONT_{R/L}_SSD{1/2/3/4} -> returns direction and number
        """
        # Method 1: Single comprehensive regex pattern
        pattern = r'^(GO|STOP|CONT)_([RL])(?:_SSD([1-4]))?$'
        match = re.match(pattern, text)
        
        if match:
            prefix = match.group(1)
            direction = match.group(2)
            ssd_number = match.group(3)  # Will be None for GO type
            
            if prefix == "GO":
                return {"type": "GO", "direction": direction}
            else:
                return {"type": prefix, "direction": direction, "ssd_number": int(ssd_number)}
        
        return None
    
    def plot_behavior(self, with_saccades=True):
        plot_line_width = 3
        saccade_line_width = 2
        time = np.arange(len(self.hPos))  # Assuming 1 ms intervals
        df = pd.DataFrame({
            'Time (ms)': time,
            'Horizontal Position (deg)': self.hPos,
            'Vertical Position (deg)': self.vPos,
            'Horizontal Velocity (deg/s)': self.hVel,
            'Vertical Velocity (deg/s)': self.vVel,
            'Speed (deg/s)': self.speed
        })
        
        # Melt the DataFrame for easier plotting with hvplot
        df_melted = df.melt(id_vars='Time (ms)', var_name='Measure', value_name='Value')
        
        # Create interactive plot with hvplot
        plot = df_melted.hvplot.line(
            x='Time (ms)', 
            y='Value', 
            by='Measure', 
            title=f'Trial: {self.trial_name} ({self.filename})',
            height=600, 
            width=800, 
            legend='bottom_left',
            line_width=plot_line_width,
            muted_alpha=0
        ).opts(
            tools=['hover'], 
            active_tools=['wheel_zoom'], 
            show_grid=True
        )

        for i in range(len(self.segs_times)-1):
            plot *= hv.VSpan(
                self.segs_times[i], self.segs_times[i+1]).opts(
                    fill_alpha=0.2   
            )
            
        if with_saccades and hasattr(self, 'saccades'):

            for start, end in self.saccades:
                plot *= hv.VLine(start).opts(color='green', line_dash='dashed', line_width=saccade_line_width, show_legend=True)
                plot *= hv.VLine(end).opts(color='red', line_dash='dashed', line_width=saccade_line_width)
        
        return plot

    def get_saccades(self, speed_threshold=20, end_buffer=15, start_buffer=20):
        # Convert the speed data to a pandas Series
        speed_series = pd.DataFrame(self.speed, columns=['speed'])

        go_no_go = speed_series.where(
            speed_series['speed'] > speed_threshold, 0
        ).mask(speed_series['speed'] > speed_threshold, 1)
        gng_arr = go_no_go.to_numpy().flatten()
        go_times = np.where(gng_arr == 1)[0]
        go_times_diff = np.diff(go_times)
        skips = np.where(go_times_diff > 1)[0] 
        try:
            saccade_ends = np.minimum(
                (np.append(go_times[skips], go_times[-1]) + end_buffer), 
                len(gng_arr)
            )
            saccade_starts = np.maximum(
                (np.append(go_times[0], go_times[skips+1]) - start_buffer), 
                0
            )
            saccades = np.dstack((saccade_starts, saccade_ends))[0]
            self.saccades = saccades
            return saccades
        except Exception as e:
            return np.array([])    
        
    def get_first_relevant_saccade(self):
        if not hasattr(self, 'saccades'):
            self.get_saccades()
        try:
            saccade_times = np.array(self.saccades)
            if saccade_times.ndim == 1:
                saccade_times = saccade_times.reshape(1,2)
            if saccade_times.shape == (2,):
                first_relevant_saccade = saccade_times
            else:
                first_saccade_idx = np.array(np.where(
                    (saccade_times[:,0] - self.go_cue) > 0
                ))[:,0]
                first_relevant_saccade = np.array(
                    saccade_times[first_saccade_idx[0]], dtype=np.int16
                )
        except:
            first_relevant_saccade = np.nan

        self.__first_relevant_saccade = first_relevant_saccade
        return first_relevant_saccade


    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a single_trial instance from a dictionary of attributes.
        The dictionary can contain any attributes normally present on an instance.
        """
        obj = cls.__new__(cls)
        # Ensure file_path exists on the instance (may be None if not provided)
        setattr(obj, 'file_path', data.get('file_path'))
        for key, value in data.items():
            setattr(obj, key, value)
        return obj


if __name__ == "__main__":
    data_dir = Path.cwd() / "data/fiona_sst/fi211109"
    file_path = data_dir / "fi211109a.2040"
    print(file_path)    

    single_trial_instance = single_trial(file_path)
    single_trial_instance.get_saccades()
    single_trial_instance.get_first_relevant_saccade()
    # pprint(single_trial_instance.saccades)
    # pprint(single_trial_instance.first_relevant_saccade)
    # Uncomment to test get_file_data separately
    # test_data = single_trial_instance.get_file_data(file_path)
    # pprint(test_data)
    # pprint(single_trial_instance.data_dict)
    # print(repr(single_trial_instance))
    # print(single_trial_instance)
    plot = single_trial_instance.plot_behavior()
        
    #%%
    show(hv.render(plot))
# %%
