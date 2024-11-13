import numpy as np
from sklearn.cluster import KMeans

def leach_kmeans(nodes, sink, energy_params, packet_size, do, rounds):
    # Variabel untuk menyimpan statistik dari simulasi
    stats = {'dead': [], 'energy': []}
    
    p = energy_params['p']

    for r in range(rounds):
        # Filter node hidup dan tentukan jumlah cluster berdasarkan persentase `p`
        alive_nodes = [(node['x'], node['y']) for node in nodes if node['energy'] > 0]
        n_clusters = max(1, int(len(alive_nodes) * p))  # Minimum satu cluster

        # Jika tidak ada node yang hidup, lanjutkan ke ronde berikutnya
        if not alive_nodes:
            dead_nodes = sum(1 for node in nodes if node['energy'] <= 0)
            remaining_energy = sum(node['energy'] for node in nodes if node['energy'] > 0)
            stats['dead'].append(dead_nodes)
            stats['energy'].append(remaining_energy)
            continue

        # Sesuaikan jumlah cluster jika node hidup tidak cukup
        if len(alive_nodes) < n_clusters:
            n_clusters = len(alive_nodes)

        # Penerapan K-Means clustering pada node yang hidup
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(alive_nodes)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        # Tentukan cluster head dengan energi tertinggi dalam setiap cluster
        cluster_heads = []
        for i in range(n_clusters):
            cluster_indices = [idx for idx, label in enumerate(labels) if label == i]
            if cluster_indices:
                head_index = cluster_indices[np.argmax([nodes[idx]['energy'] for idx in cluster_indices])]
                cluster_heads.append(head_index)

        # Hitung konsumsi energi untuk komunikasi
        for i, node in enumerate(nodes):
            if node['energy'] > 0:
                if i in cluster_heads:
                    # Cluster head mengirim ke sink
                    dist_to_sink = np.linalg.norm([node['x'] - sink['x'], node['y'] - sink['y']])
                    if dist_to_sink > do:
                        energy_cost = (energy_params['E_tx'] * packet_size +
                                       energy_params['E_mp'] * packet_size * (dist_to_sink ** 4))
                    else:
                        energy_cost = (energy_params['E_tx'] * packet_size +
                                       energy_params['E_fs'] * packet_size * (dist_to_sink ** 2))
                    node['energy'] = max(node['energy'] - energy_cost, 0)
                else:
                    # Node non-cluster head mengirim ke cluster head terdekat
                    distances = [np.linalg.norm([node['x'] - nodes[ch]['x'], node['y'] - nodes[ch]['y']]) for ch in cluster_heads]
                    nearest_ch = cluster_heads[np.argmin(distances)]
                    nearest_dist = distances[np.argmin(distances)]

                    if nearest_dist > do:
                        energy_cost = (energy_params['E_tx'] * packet_size +
                                       energy_params['E_mp'] * packet_size * (nearest_dist ** 4))
                    else:
                        energy_cost = (energy_params['E_tx'] * packet_size +
                                       energy_params['E_fs'] * packet_size * (nearest_dist ** 2))
                    node['energy'] = max(node['energy'] - energy_cost, 0)

                    # Cluster head menerima data dari node biasa
                    nodes[nearest_ch]['energy'] = max(nodes[nearest_ch]['energy'] - energy_params['E_elec'] * packet_size, 0)

        # Simpan statistik setiap ronde
        dead_nodes = sum(1 for node in nodes if node['energy'] <= 0)
        remaining_energy = sum(node['energy'] for node in nodes if node['energy'] > 0)
        stats['dead'].append(dead_nodes)
        stats['energy'].append(remaining_energy)

    return stats
