import numpy as np

# Fungsi LEACH Original dengan Koreksi
def leach_ori(nodes, sink, energy_params, packet_size, do, rounds, cycle_length):
    n = len(nodes)
    stats = {'dead': [], 'energy': [], 'cluster_heads': []}
    
    # Siklus threshold untuk pemilihan Cluster Head
    def threshold(r, node_id):
        p = energy_params['p']
        if (r % cycle_length == 0):
            return p / (1 - p * (r % cycle_length))
        else:
            return 0

    for r in range(rounds):
        # Reset cluster heads setiap siklus
        cluster_heads = []
        remaining_energy = sum([node['energy'] for node in nodes if node['energy'] > 0])

        # Seleksi Cluster Heads berbasis threshold dan energi residual
        for i, node in enumerate(nodes):
            if node['energy'] > 0 and np.random.rand() < threshold(r, i):  # threshold for CH selection
                cluster_heads.append(i)

        # Jika tidak ada Cluster Head yang terpilih, pilih satu dengan energi tertinggi
        if not cluster_heads and any(node['energy'] > 0 for node in nodes):
            max_energy_node = np.argmax([node['energy'] for node in nodes])
            cluster_heads.append(max_energy_node)

        # Pembentukan cluster: menetapkan node ke Cluster Head terdekat atau sink jika lebih efisien
        for i, node in enumerate(nodes):
            if i not in cluster_heads and node['energy'] > 0:
                if cluster_heads:  # Pastikan ada cluster head
                    # Temukan cluster head terdekat
                    distances = [np.linalg.norm([nodes[ch]['x'] - node['x'], nodes[ch]['y'] - node['y']]) for ch in cluster_heads]
                    nearest_ch = cluster_heads[np.argmin(distances)]
                    distance_to_ch = distances[np.argmin(distances)]
                    
                    # Pilih untuk mengirim langsung ke sink jika lebih hemat energi
                    distance_to_sink = np.linalg.norm([node['x'] - sink['x'], node['y'] - sink['y']])
                    if distance_to_sink < distance_to_ch:
                        # Energi untuk pengiriman langsung ke sink
                        if distance_to_sink > do:
                            energy_cost = (energy_params['E_tx'] * packet_size + 
                                           energy_params['E_mp'] * packet_size * (distance_to_sink ** 4))
                        else:
                            energy_cost = (energy_params['E_tx'] * packet_size + 
                                           energy_params['E_fs'] * packet_size * (distance_to_sink ** 2))
                    else:
                        # Energi untuk pengiriman ke Cluster Head
                        if distance_to_ch > do:
                            energy_cost = (energy_params['E_tx'] * packet_size + 
                                           energy_params['E_mp'] * packet_size * (distance_to_ch ** 4))
                        else:
                            energy_cost = (energy_params['E_tx'] * packet_size + 
                                           energy_params['E_fs'] * packet_size * (distance_to_ch ** 2))
                        
                        # Cluster Head juga mengonsumsi energi saat menerima pesan
                        nodes[nearest_ch]['energy'] = max(nodes[nearest_ch]['energy'] - energy_params['E_elec'] * packet_size, 0)
                    
                    node['energy'] = max(node['energy'] - energy_cost, 0)

        # Update energi untuk Cluster Head yang mengirim data ke sink
        for ch in cluster_heads:
            distance_to_sink = np.linalg.norm([nodes[ch]['x'] - sink['x'], nodes[ch]['y'] - sink['y']])
            if distance_to_sink > do:
                energy_cost = energy_params['E_tx'] * packet_size + energy_params['E_mp'] * packet_size * (distance_to_sink ** 4)
            else:
                energy_cost = energy_params['E_tx'] * packet_size + energy_params['E_fs'] * packet_size * (distance_to_sink ** 2)
            
            nodes[ch]['energy'] = max(nodes[ch]['energy'] - energy_cost, 0)

        # Track statistik
        dead_nodes = sum([1 for node in nodes if node['energy'] <= 0])
        stats['dead'].append(dead_nodes)
        stats['energy'].append(remaining_energy)
        stats['cluster_heads'].append(len(cluster_heads))

    return stats
