import util as u

def add_button_to(plt, fn):
    #print(fn)
    from matplotlib.widgets import Button
    ax = plt.axes([0.85, 0.95, 0.15, 0.03])
    btn = Button(ax, 'New')
    btn.on_clicked(fn)