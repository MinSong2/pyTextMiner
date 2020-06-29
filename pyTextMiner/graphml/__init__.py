import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import platform
from matplotlib.ft2font import FT2Font
import matplotlib as mpl

class GraphMLCreator:

    def __init__(self):
        self.G = nx.Graph()

        # Hack: offset the most central node to avoid too much overlap
        self.rad0 = 0.3

    def createGraphML(self, co_occurrence, vocabulary, file):
        G = nx.Graph()

        for obj in vocabulary:
            G.add_node(obj)
        # convert list to a single dictionary

        for pair in co_occurrence:
            node1 = ''
            node2 = ''
            for inner_pair in pair:

                if type(inner_pair) is tuple:
                    node1 = inner_pair[0]
                    node2 = inner_pair[1]
                elif type(inner_pair) is str:
                    inner_pair=inner_pair.split()
                    node1 = inner_pair[0]
                    node2 = inner_pair[1]
                elif type(inner_pair) is int:
                    #print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : " + str(tuple[node1]))
                    G.add_edge(node1, node2, weight=float(inner_pair))
                elif type(inner_pair) is float:
                    #print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : ")
                    G.add_edge(node1, node2, weight=float(inner_pair))

        self.G = G
        print(self.G.number_of_nodes())
        nx.write_graphml(G, file)

    def createGraphMLWithThreshold(self, co_occurrence, word_hist, vocab, file, threshold=10.0):
        G = nx.Graph()

        filtered_word_list=[]
        for pair in co_occurrence:
            node1 = ''
            node2 = ''
            for inner_pair in pair:
                if type(inner_pair) is tuple:
                    node1 = inner_pair[0]
                    node2 = inner_pair[1]
                elif type(inner_pair) is str:
                    inner_pair=inner_pair.split()
                    node1 = inner_pair[0]
                    node2 = inner_pair[1]
                elif type(inner_pair) is int:
                    if float(inner_pair) >= threshold:
                        #print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : " + str(tuple[node1]))
                        G.add_edge(node1, node2, weight=float(inner_pair))
                        if node1 not in filtered_word_list:
                            filtered_word_list.append(node1)
                        if node2 not in filtered_word_list:
                            filtered_word_list.append(node2)
                elif type(inner_pair) is float:
                    if float(inner_pair) >= threshold:
                        #print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : ")
                        G.add_edge(node1, node2, weight=float(inner_pair))
                        if node1 not in filtered_word_list:
                            filtered_word_list.append(node1)
                        if node2 not in filtered_word_list:
                            filtered_word_list.append(node2)
        for word in word_hist:
            if word in filtered_word_list:
                G.add_node(word, count=word_hist[word])

        self.G = G
        print(self.G.number_of_nodes())
        nx.write_graphml(G, file)

    def createGraphMLWithThresholdInDictionary(self, co_occurrence, word_hist, file, threshold=10.0):
        G = nx.Graph()

        node1 = ''
        node2 = ''
        filtered_word_list=[]
        for pair in co_occurrence:
            node1 = str(pair[0])
            node2 = str(pair[1])

            inner_pair = co_occurrence[pair]

            if type(inner_pair) is str:
                inner_pair=inner_pair.split()
                node1 = inner_pair[0]
                node2 = inner_pair[1]
            elif type(inner_pair) is int:
                if float(inner_pair) >= threshold:
                    #print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : " + str(tuple[node1]))
                    G.add_edge(node1, node2, weight=float(inner_pair))
                    if node1 not in filtered_word_list:
                        filtered_word_list.append(node1)
                    if node2 not in filtered_word_list:
                        filtered_word_list.append(node2)
            elif type(inner_pair) is float:
                if float(inner_pair) >= threshold:
                    #print ("X " + node1 + " == " + node2 + " == " + str(inner_pair) + " : ")
                    G.add_edge(node1, node2, weight=float(inner_pair))
                    if node1 not in filtered_word_list:
                        filtered_word_list.append(node1)
                    if node2 not in filtered_word_list:
                        filtered_word_list.append(node2)

        for word in word_hist:
            if word in filtered_word_list:
                G.add_node(word, count=word_hist[word])

        self.G = G
        print(self.G.number_of_nodes())
        nx.write_graphml(G, file)

    def centrality_layout(self):
        centrality = nx.eigenvector_centrality_numpy(self.G)
        """Compute a layout based on centrality.
        """
        # Create a list of centralities, sorted by centrality value
        cent = sorted(centrality.items(), key=lambda x:float(x[1]), reverse=True)
        nodes = [c[0] for c in cent]
        cent  = np.array([float(c[1]) for c in cent])
        rad = (cent - cent[0])/(cent[-1]-cent[0])
        rad = self.rescale_arr(rad, self.rad0, 1)
        angles = np.linspace(0, 2*np.pi, len(centrality))
        layout = {}
        for n, node in enumerate(nodes):
            r = rad[n]
            th = angles[n]
            layout[node] = r*np.cos(th), r*np.sin(th)
        return layout

    def plot_graph(self, title=None, file='graph.png'):
        from matplotlib.font_manager import _rebuild
        _rebuild()

        font_path = ''
        if platform.system() is 'Windows':
            # Window의 경우 폰트 경로
            font_path = 'C:/Windows/Fonts/malgun.ttf'
        elif platform.system() is 'Darwin':
            # for Mac
            font_path='/Library/Fonts/AppleGothic.ttf'

        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)
        # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
        mpl.rcParams['axes.unicode_minus'] = False
        #print('버전: ', mpl.__version__)
        #print('설치 위치: ', mpl.__file__)
        #print('설정 위치: ', mpl.get_configdir())
        #print('캐시 위치: ', mpl.get_cachedir())

        # size, family
        print('# 설정 되어있는 폰트 사이즈')
        print(plt.rcParams['font.size'])
        print('# 설정 되어있는 폰트 글꼴')
        print(plt.rcParams['font.family'])

        fig = plt.figure(figsize=(8, 8))
        pos = self.centrality_layout()

        """Conveniently summarize graph visually"""
        # config parameters
        edge_min_width= 3
        edge_max_width= 12
        label_font = 18
        node_font = 22
        node_alpha = 0.4
        edge_alpha = 0.55
        edge_cmap = plt.cm.Spectral

        # Create figure
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.add_subplot(111)
        fig.subplots_adjust(0,0,1)

        font = FT2Font(font_path)

        # Plot nodes with size according to count
        sizes = []
        degrees = []
        for n, d in self.G.nodes(data=True):
            sizes.append(d['count'])
            degrees.append(self.G.degree(n))

        sizes = self.rescale_arr(np.array(sizes, dtype=float), 100, 1000)

        # Compute layout and label edges according to weight
        pos = nx.spring_layout(self.G) if pos is None else pos
        labels = {}
        width = []
        for n1, n2, d in self.G.edges(data=True):
            w = d['weight']
            labels[n1, n2] = w
            width.append(w)

        width = self.rescale_arr(np.array(width, dtype=float), edge_min_width,
                            edge_max_width)

        # Draw
        nx.draw_networkx_nodes(self.G, pos, node_size=sizes, node_color=degrees,
                               alpha=node_alpha)
        nx.draw_networkx_edges(self.G, pos, width=width, edge_color=width,
                               edge_cmap=edge_cmap, alpha=edge_alpha)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels,
                                     font_size=label_font)
        nx.draw_networkx_labels(self.G, pos, font_size=node_font, font_family=font_name, font_weight='bold')

        if title is not None:
            ax.set_title(title, fontsize=label_font)
        ax.set_xticks([])
        ax.set_yticks([])

        # Mark centrality axes
        kw = dict(color='k', linestyle='-')
        cross = [ax.axhline(0, **kw), ax.axvline(self.rad0, **kw)]
        [l.set_zorder(0) for l in cross]

        #plt.show()
        plt.savefig(file)

    def rescale_arr(self, arr, amin, amax):
        """Rescale an array to a new range.
        Return a new array whose range of values is (amin, amax).
        Parameters
        ----------
        arr : array-like
        amin : float
          new minimum value
        amax : float
          new maximum value
        Examples
        --------
        #>>> a = np.arange(5)
        #>>> rescale_arr(a,3,6)
        array([ 3.  ,  3.75,  4.5 ,  5.25,  6.  ])
        """

        # old bounds
        m = arr.min()
        M = arr.max()
        # scale/offset
        s = float(amax - amin) / (M - m)
        d = amin - s * m

        # Apply clip before returning to cut off possible overflows outside the
        # intended range due to roundoff error, so that we can absolutely guarantee
        # that on output, there are no values > amax or < amin.
        return np.clip(s * arr + d, amin, amax)

    def summarize_centrality(self, limit=10):

        centrality = nx.eigenvector_centrality_numpy(self.G)
        c = centrality.items()
        c = sorted(c, key=lambda x: x[1], reverse=True)
        print('\nGraph centrality')
        count=0
        for node, cent in c:
            if count>limit:
                break
            print ("%15s: %.3g" % (node, float(cent)))
            count+=1

    def sort_freqs(self, freqs):
        """Sort a word frequency histogram represented as a dictionary.
        Parameters
        ----------
        freqs : dict
          A dict with string keys and integer values.
        Return
        ------
        items : list
          A list of (count, word) pairs.
        """
        items = freqs.items()
        items.sort(key=lambda wc: wc[1])
        return items

    def plot_word_histogram(self, freqs, show=10, title=None):
        """Plot a histogram of word frequencies, limited to the top `show` ones.
        """
        sorted_f = self.sort_freqs(freqs) if isinstance(freqs, dict) else freqs

        # Don't show the tail
        if isinstance(show, int):
            # interpret as number of words to show in histogram
            show_f = sorted_f[-show:]
        else:
            # interpret as a fraction
            start = -int(round(show * len(freqs)))
            show_f = sorted_f[start:]

        # Now, extract words and counts, plot
        n_words = len(show_f)
        ind = np.arange(n_words)
        words = [i[0] for i in show_f]
        counts = [i[1] for i in show_f]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if n_words <= 20:
            # Only show bars and x labels for small histograms, they don't make
            # sense otherwise
            ax.bar(ind, counts)
            ax.set_xticks(ind)
            ax.set_xticklabels(words, rotation=45)
            fig.subplots_adjust(bottom=0.25)
        else:
            # For larger ones, do a step plot
            ax.step(ind, counts)

        # If it spans more than two decades, use a log scale
        if float(max(counts)) / min(counts) > 100:
            ax.set_yscale('log')

        if title:
            ax.set_title(title)
        return ax