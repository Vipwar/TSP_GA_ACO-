import csv
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from utils.tsp_utils import generate_cities
from algorithms.ga import GA
from algorithms.aco import ACO

class TSPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP Solver – GA vs ACO")
        self.resize(1100, 600)

        self.setStyleSheet("""
            QWidget { background:#121212; color:white; font-size:13px; }
            QPushButton { background:#1f6feb; padding:8px; border-radius:6px; }
            QPushButton:hover { background:#388bfd; }
            QGroupBox { border:1px solid #333; border-radius:6px; margin-top:10px; }
        """)

        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)

        # CONTROL
        left = QVBoxLayout()
        box = QGroupBox("Cấu hình")
        form = QFormLayout()

        self.city_spin = QSpinBox()
        self.city_spin.setRange(5, 50)
        self.city_spin.setValue(15)

        self.run_btn = QPushButton("Chạy GA & ACO")
        self.run_btn.clicked.connect(self.run)

        self.save_btn = QPushButton("Xuất CSV")
        self.save_btn.clicked.connect(self.save_csv)

        form.addRow("Số thành phố:", self.city_spin)
        box.setLayout(form)

        left.addWidget(box)
        left.addWidget(self.run_btn)
        left.addWidget(self.save_btn)
        left.addStretch()

        # GRAPH
        self.fig_ga = Figure(facecolor="#1e1e1e")
        self.fig_aco = Figure(facecolor="#1e1e1e")

        self.can_ga = FigureCanvasQTAgg(self.fig_ga)
        self.can_aco = FigureCanvasQTAgg(self.fig_aco)

        graphs = QHBoxLayout()
        graphs.addWidget(self.can_ga)
        graphs.addWidget(self.can_aco)

        main.addLayout(left, 1)
        main.addLayout(graphs, 4)

    def run(self):
        self.cities = generate_cities(self.city_spin.value())

        self.ga = GA(self.cities)
        self.aco = ACO(self.cities)

        self.ga_gen = self.ga.run_stepwise()
        self.aco_gen = self.aco.run_stepwise()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(300)

    def update(self):
        try:
            g_i, g_path, g_d = next(self.ga_gen)
            a_i, a_path, a_d = next(self.aco_gen)

            self.draw(self.fig_ga, self.can_ga, g_path, g_d, f"GA – Gen {g_i}")
            self.draw(self.fig_aco, self.can_aco, a_path, a_d, f"ACO – Iter {a_i}")
        except StopIteration:
            self.timer.stop()

    def draw(self, fig, canvas, path, dist, title):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e")

        xs = [self.cities[i][0] for i in path] + [self.cities[path[0]][0]]
        ys = [self.cities[i][1] for i in path] + [self.cities[path[0]][1]]

        ax.plot(xs, ys, "o-", color="#00d4ff", linewidth=2)
        ax.set_title(f"{title}\nDistance = {dist:.2f}", color="white")
        ax.tick_params(colors="white")

        canvas.draw()

    def save_csv(self):
        with open("ga_history.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "BestDistance"])
            writer.writerows(self.ga.history)

        with open("aco_history.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "BestDistance"])
            writer.writerows(self.aco.history)
