import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import umap


class Visualizer:
    def plot_axis_of_controversy(self, party_df: pd.DataFrame, issue: str, anchors=None):
        show_anchors = anchors is not None
        fig, ax = plt.subplots(figsize=(14, 5 if show_anchors else 3.5))

        df = party_df.sort_values('controversy_score').reset_index(drop=True)

        unique_parties = sorted(df['party'].unique())
        cmap = plt.get_cmap('tab10')
        party_to_color = {p: cmap(i % 10) for i, p in enumerate(unique_parties)}
        colors = [party_to_color[p] for p in df['party']]

        x_min, x_max = df['controversy_score'].min(), df['controversy_score'].max()
        pad = max((x_max - x_min) * 0.25, 0.05)

        ax.axhline(0, color='#444444', linewidth=2, zorder=2)
        ax.scatter(df['controversy_score'], np.zeros(len(df)), c=colors, s=150, zorder=5, linewidths=0.8, edgecolors='white')

        stagger_heights = [0.10, -0.14, 0.19, -0.24]
        for i, (_, row) in enumerate(df.iterrows()):
            yo = stagger_heights[i % len(stagger_heights)]
            c = party_to_color[row['party']]
            ax.annotate(
                row['party'],
                xy=(row['controversy_score'], 0.01 if yo > 0 else -0.01),
                xytext=(row['controversy_score'], yo),
                ha='center', va='center',
                fontsize=8.5, color=c, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=c, lw=0.8, alpha=0.5),
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
            )

        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.text(0, -0.34, 'neutral', ha='center', fontsize=7, color='gray', style='italic')
        ax.text(x_max + pad * 0.75, 0, 'PRO ▶', ha='left', fontsize=10, color='steelblue', fontweight='bold', va='center')
        ax.text(x_min - pad * 0.75, 0, '◀ CON', ha='right', fontsize=10, color='firebrick', fontweight='bold', va='center')

        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(-0.40, 0.32)
        ax.set_title(f"Axis of Controversy: {issue}", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Controversy Score", fontsize=9, color='#666666')
        ax.set_yticks([])
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.grid(axis='x', alpha=0.2)

        if show_anchors:
            plt.subplots_adjust(bottom=0.38)
            fig.text(0.05, 0.24, f"PRO:  {anchors['pro']}", ha='left', va='bottom', fontsize=8, color='steelblue', style='italic')
            fig.text(0.05, 0.08, f"CON:  {anchors['con']}", ha='left', va='bottom', fontsize=8, color='firebrick', style='italic')

        plt.show()

    def compute_umap_embeddings(self, sum_df: pd.DataFrame, speaker_embeddings, anchor_embeddings,
                                n_components=2, n_neighbors=10, min_dist=0.1, metric="cosine",
                                random_seed=42, anchors=None) -> dict:
        embeddings = np.vstack([speaker_embeddings, anchor_embeddings])

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_seed
        )
        reduced_embeddings = reducer.fit_transform(embeddings)

        n_speakers = len(sum_df)
        reduced_speeches = reduced_embeddings[:n_speakers]
        reduced_anchors = reduced_embeddings[n_speakers:]

        df = sum_df.copy()
        df["umap_x"] = reduced_speeches[:, 0]
        df["umap_y"] = reduced_speeches[:, 1]

        return {
            'df': df,
            'reduced_embeddings': reduced_embeddings,
            'reduced_anchors': reduced_anchors,
            'anchors': anchors
        }

    def plot_umap_party_averages(self, umap_data: dict, show_speeches=True, show_party_averages=True,
                                 show_speaker_labels=True, label_fontsize=8):
        sum_df = umap_data['df']
        reduced_anchors = umap_data['reduced_anchors']
        anchors = umap_data['anchors']

        party_centroids = (
            sum_df
            .groupby("party")
            .agg({"umap_x": "mean", "umap_y": "mean", "party": "size"})
            .rename(columns={"party": "speech_count"})
            .reset_index()
        )

        unique_parties = sorted(sum_df["party"].unique())
        party_to_color = {party: i for i, party in enumerate(unique_parties)}
        cmap = plt.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=(12, 12))

        for party in unique_parties:
            color = cmap(party_to_color[party])

            if show_speeches:
                mask = sum_df["party"] == party
                party_data = sum_df.loc[mask]
                ax.scatter(party_data["umap_x"], party_data["umap_y"], alpha=0.85, color=color, label=party)

                if show_speaker_labels:
                    for _, row in party_data.iterrows():
                        ax.annotate(
                            row['speaker'],
                            xy=(row['umap_x'], row['umap_y']),
                            xytext=(3, 3),
                            textcoords='offset points',
                            fontsize=label_fontsize,
                            alpha=0.7,
                            color=color
                        )

            if show_party_averages:
                centroid = party_centroids[party_centroids["party"] == party]
                marker_size = 100 + centroid["speech_count"].values[0] * 80
                ax.scatter(centroid["umap_x"], centroid["umap_y"], s=marker_size, color=color,
                           edgecolor="black", linewidth=1.2, zorder=5)

        if reduced_anchors is not None:
            ax.scatter(reduced_anchors[:, 0], reduced_anchors[:, 1], s=200, color='black',
                       marker='D', edgecolor='white', linewidth=1.5, label='Stance Anchors', zorder=10)

            for x, y, label in zip(reduced_anchors[:, 0], reduced_anchors[:, 1], ['pro', 'con']):
                ax.annotate(
                    label.upper(),
                    xy=(x, y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1)
                )

        ax.set_title(f"UMAP Party Averages – {anchors['topic']}")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.legend(title="Parties & Anchors", bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.text(
            0.02, 0.02,
            f"*PRO*: {anchors['pro']}\n\n*CON*: {anchors['con']}",
            ha="left", va="bottom", fontsize=14, wrap=True
        )
        plt.subplots_adjust(bottom=0.30)
        plt.show()
