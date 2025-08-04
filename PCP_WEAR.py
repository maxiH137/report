import pandas as pd
import tkinter as tk
import random
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram


def load_csv(filename):
    return pd.read_csv(filename)

def get_color(idx):
    random.seed(idx)
    return "#%06x" % random.randint(0, 0xFFFFFF)

def select_all_activities():
    for var in visible_flags.values():
        var.set(True)
    draw_plot()

def unselect_all_activities():
    for var in visible_flags.values():
        var.set(False)
    draw_plot()

def rename_file(file):
    for i, name in enumerate(['ArmR_X', 'ArmR_Y', 'ArmR_Z', 'LegR_X', 'LegR_Y', 'LegR_Z',
                              'LegL_X', 'LegL_Y', 'LegL_Z', 'ArmL_X', 'ArmL_Y', 'ArmL_Z']):
        file = file.rename(columns={file.columns[i]: name})
    return file

# UI State
groups = {}
visible_flags = {}
checkbuttons = {}
clickable_points = []
features = []

# Tkinter Setup
root = tk.Tk()
root.title("Parallel Coordinates Plot")

show_average_var = tk.BooleanVar(master=root, value=False)
use_graph_analysis_var = tk.BooleanVar(master=root, value=False)
normalize_var = tk.BooleanVar(master=root, value=True)

canvas_width = 1000
canvas_height = 600
margin = 80
axis_height = canvas_height - 2 * margin

# Layout frames
main_frame = tk.Frame(root)
main_frame.pack()

left_frame = tk.Frame(main_frame)
left_frame.grid(row=0, column=0, padx=5)

right_frame = tk.Frame(main_frame)
right_frame.grid(row=0, column=1, sticky='n')

# Subject dropdown
dropdown_sbj = tk.StringVar(master=root)
dropdown_sbj.set("0")

def on_subject_change(*args):
    subject_id = int(dropdown_sbj.get())
    process_subject(subject_id)

dropdown_sbj.trace('w', on_subject_change)

subject_label = tk.Label(left_frame, text="Subject:")
subject_label.pack(anchor='w', padx=10, pady=(10, 0))
dropdown_menu = tk.OptionMenu(left_frame, dropdown_sbj, *list(range(18)))
dropdown_menu.pack(anchor='w', padx=10, pady=(0, 10))

# Sample size
sample_input_label = tk.Label(left_frame, text="Sample Size:")
sample_input_label.pack(anchor='w', padx=10)
sample_input = tk.Entry(left_frame, width=30)
sample_input.insert(0, "1000")
sample_input.pack(anchor='w', padx=10, pady=(0, 10))

# Canvas
canvas = tk.Canvas(left_frame, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()
canvas.bind("<Button-1>", lambda e: on_canvas_click(e))

# Legend
legend_label = tk.Label(right_frame, text="Activities", font=("Arial", 12, "bold"))
legend_label.pack()
btn_frame = tk.Frame(right_frame)
btn_frame.pack(pady=(5, 10))
tk.Button(btn_frame, text="Select All", command=select_all_activities).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Unselect All", command=unselect_all_activities).grid(row=0, column=1, padx=5)

avg_checkbox = tk.Checkbutton(right_frame, text="Show Average Lines", variable=show_average_var, command=lambda: draw_plot())
avg_checkbox.pack(pady=5)

normalize_checkbox = tk.Checkbutton(right_frame, text="Normalize Data", variable=normalize_var, command=lambda: process_subject(int(dropdown_sbj.get())))
normalize_checkbox.pack(pady=5)

graph_checkbox = tk.Checkbutton(right_frame, text="Use Graph Analysis", variable=use_graph_analysis_var, command=lambda: process_subject(int(dropdown_sbj.get())))
graph_checkbox.pack(pady=5)

legend_canvas = tk.Canvas(right_frame, width=300, height=canvas_height)
scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=legend_canvas.yview)
scrollable_frame = tk.Frame(legend_canvas)

scrollable_frame.bind("<Configure>", lambda e: legend_canvas.configure(scrollregion=legend_canvas.bbox("all")))
legend_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
legend_canvas.configure(yscrollcommand=scrollbar.set)
legend_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def process_subject(subject_id):
    global file, groups, visible_flags, checkbuttons, features

    file = load_csv(f'data/raw/inertial/sbj_{subject_id}.csv')
    file = file.iloc[:, 1:]
    file = rename_file(file)
    file = file.fillna(0)

    sample = sample_input.get()
    if not sample.isdigit() or int(sample) <= 0:
        sample = 1000

    file = file.sample(n=min(int(sample), len(file)), random_state=42)
    file.reset_index(drop=False, inplace=True)
    file.rename(columns={'index': 'original_index'}, inplace=True)
    # file.rename(columns={'index': 'original_index'}, inplace=True)  # Handled manually above

    if normalize_var.get():
        for col in file.columns:
            if col not in ['original_index', file.columns[-1]]:
                file[col] = (file[col] - file[col].min()) / (file[col].max() - file[col].min())


    groups = {label: group.reset_index(drop=True) for label, group in file.groupby(file.columns[-1])}

    raw_features = [col for col in file.columns if col not in ['original_index', file.columns[-1]]]

    if use_graph_analysis_var.get():
        feature_data = file[raw_features]
        corr_matrix = feature_data.corr().fillna(0)
        similarity = np.abs(corr_matrix)

        link = linkage(1 - similarity, method='average')
        dendro = dendrogram(link, labels=similarity.columns, no_plot=True)
        ordered = dendro['ivl']
    else:
        ordered = raw_features

    features[:] = ordered

    for cb in checkbuttons.values():
        cb.pack_forget()

    for idx, label in enumerate(groups.keys()):
        if label not in visible_flags:
            visible_flags[label] = tk.BooleanVar(value=True)
        if label not in checkbuttons:
            cb = tk.Checkbutton(scrollable_frame, text=label, variable=visible_flags[label], command=draw_plot, anchor='w', width=25)
            checkbuttons[label] = cb
        checkbuttons[label].pack(anchor='w', pady=1)

    draw_plot()

def draw_plot():
    canvas.delete("all")
    clickable_points.clear()

    tick_count = 5
    tick_spacing = axis_height / tick_count
    axis_spacing = (canvas_width - 2 * margin) / (len(features) - 1)

    for i, feat in enumerate(features):
        x = margin + i * axis_spacing
        canvas.create_line(x, margin, x, margin + axis_height, fill='black')
        canvas.create_text(x, margin - 20, text=feat, angle=45, anchor='w')

        col_values = file[feat]
        col_min, col_max = col_values.min(), col_values.max()

        for t in range(tick_count + 1):
            y = margin + t * tick_spacing
            if normalize_var.get():
                value = 100 - t * 20
                canvas.create_line(x - 5, y, x + 5, y, fill='black')
                canvas.create_text(x - 10, y, text=str(int(value)), anchor='e')
            else:
                value = col_min + (col_max - col_min) * (1 - t / tick_count)
                canvas.create_line(x - 5, y, x + 5, y, fill='black')
                canvas.create_text(x - 10, y, text=f"{value:.1f}", anchor='e')

    for idx, (label, group) in enumerate(groups.items()):
        if not visible_flags[label].get():
            continue
        color = get_color(idx)
        for _, row in group.iterrows():
            points = []
            for i, feat in enumerate(features):
                x = margin + i * axis_spacing
                if normalize_var.get():
                    y = margin + axis_height - row[feat] * axis_height
                else:
                    col_min, col_max = file[feat].min(), file[feat].max()
                    y = margin + axis_height * (1 - (row[feat] - col_min) / (col_max - col_min))
                points.append((x, y))
            for j in range(len(points) - 1):
                canvas.create_line(points[j][0], points[j][1], points[j+1][0], points[j+1][1], fill=color, width=1)
            clickable_points.append({
                "points": points,
                "row": row,
                "label": label
            })

    if show_average_var.get():
        for idx, (label, group) in enumerate(groups.items()):
            if not visible_flags[label].get():
                continue
            avg_points = []
            for i, feat in enumerate(features):
                mean_val = group[feat].mean()
                x = margin + i * axis_spacing
                if normalize_var.get():
                    y = margin + axis_height - mean_val * axis_height
                else:
                    col_min, col_max = file[feat].min(), file[feat].max()
                    y = margin + axis_height * (1 - (mean_val - col_min) / (col_max - col_min))
                avg_points.append((x, y))
            for j in range(len(avg_points) - 1):
                canvas.create_line(
                    avg_points[j][0], avg_points[j][1],
                    avg_points[j+1][0], avg_points[j+1][1],
                    fill="black", width=4, dash=(6, 3)
                )

def on_canvas_click(event):
    closest = None
    min_dist = 10
    for data in clickable_points:
        for i, (x, y) in enumerate(data["points"]):
            dist = ((x - event.x) ** 2 + (y - event.y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest = {
                    "x": x,
                    "y": y,
                    "feature": features[i],
                    "value": data["row"][features[i]],
                    "index": int(data["row"]["original_index"]),
                    "label": data["label"]
                }

    if closest:
        popup = tk.Toplevel(root)
        popup.title("Data Point Info")
        info = f"""
Activity: {closest['label']}
Feature: {closest['feature']}
Value: {closest['value']:.4f}
Original Index: {closest['index']}
"""
        tk.Label(popup, text=info.strip(), justify='center', font=("Arial", 11)).pack(padx=10, pady=10)
        popup.geometry("+%d+%d" % (root.winfo_rootx() + root.winfo_width() // 2 - 100, root.winfo_rooty() + 50))
        popup.after(5000, popup.destroy)

# Load first subject
process_subject(0)
root.mainloop()
