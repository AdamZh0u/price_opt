"""mpl setup"""
import matplotlib as mpl 

def setup_mpl(as_default=1, font='en'):  # My mpl setup
    if as_default:
        mpl.rcdefaults()  # set as defult
    else:
        # Figsize
        # mpl.rcParams["figure.figsize"] = (3.60236*2, 3.5)
        # 1 col : 89 mm | 2 col : 183 mm

        # FONT
        mpl.rcParams['font.family'] = 'sans-serif'  # sans-serif
        mpl.rcParams["font.size"] = 7  # 10 default
        mpl.rcParams['legend.fontsize'] = 'small'  # medium

        # TICK
        mpl.rcParams['xtick.labelsize'] = 'small'  # medium
        mpl.rcParams['ytick.labelsize'] = 'small'  # medium
        mpl.rcParams['xtick.major.width'] = 2/3.  # 0.8
        mpl.rcParams['ytick.major.width'] = 2/3.
        mpl.rcParams['xtick.minor.width'] = 2/3.  # 0.6
        mpl.rcParams['ytick.minor.width'] = 2/3.
        mpl.rcParams['xtick.major.size'] = 3  # 3.5
        mpl.rcParams['ytick.major.size'] = 3
        mpl.rcParams['xtick.minor.size'] = 1.5  # 2
        mpl.rcParams['ytick.minor.size'] = 1.5
        mpl.rcParams['xtick.major.pad'] = '2.3'  # 3.5
        mpl.rcParams['ytick.major.pad'] = '2.3'
        mpl.rcParams['xtick.minor.pad'] = '2.3'  # 3.5
        mpl.rcParams['ytick.minor.pad'] = '2.3'
        mpl.rcParams['ytick.direction'] = 'in'  # out
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['xtick.top'] = True
        mpl.rcParams['ytick.right'] = True

        #
        mpl.rcParams['axes.linewidth'] = 2/3.  # 0.8
        mpl.rcParams['axes.labelpad'] = 2  # 4
        mpl.rcParams['lines.linewidth'] = 1  # 1.5
        mpl.rcParams['mathtext.default'] = 'regular'

        # EXPORT
        # mpl.rcParams['figure.dpi'] = 400  # 100 
        mpl.rcParams['svg.fonttype'] = "none"
        mpl.rcParams['figure.autolayout'] = True  # tight_layout

    if font == 'cn':
        print(mpl.matplotlib_fname())  # 查找字体配置路径
        print(mpl.get_cachedir())  # 查找字体缓存路径
        mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
        mpl.rcParams["axes.unicode_minus"] = False

    else:
        mpl.rcParams["font.sans-serif"] = ["Arial"]  # 设置字体

    # rasterize figure
    def save_pdf(path, fig):
        fig.savefig(path, transparent=True)

    def save_png(path, fig):
        fig.savefig(path, transparent=True)

    # vectorize figure
    def save_svg(path, fig):
        fig.savefig(path, transparent=True)

    # PARAMS
    PARAMS = {
        'mm_to_inch': 10/2.54,
        'one_col_mm': 89, # default one column figure width
        'two_col_mm': 183,
        'height_mm': 89, 
        'alpha': 0.6,
        'to_rgba': mpl.colors.ColorConverter().to_rgba,
        'save_pdf': save_pdf,
        'save_svg': save_svg,
        'save_png': save_png,
    }
    return PARAMS