import numpy as np

def leach_c(nodes, sink, energy_params, packet_size, do, rounds):
    stats = {
        'round': [],
        'alive': [],
        'dead': [],
        'cluster_heads': [],
        'energy': []
    }
    
    p = energy_params['p']
    
    n = len(nodes)
    num_ch = int(n * p)  # Menentukan jumlah ideal CH pada setiap siklus (biasanya sekitar 5-10% node)
    
    for r in range(rounds):
        # Step 1: Menentukan centroid dari node-node yang masih hidup
        alive_nodes = [node for node in nodes if node['energy'] > 0]
        avg_x = np.mean([node['x'] for node in alive_nodes])
        avg_y = np.mean([node['y'] for node in alive_nodes])
        
        # Step 2: Pemilihan Cluster Head berdasarkan jarak ke centroid dan energi tersisa
        # Urutkan node berdasarkan energi tersisa, lalu pilih `num_ch` node yang paling dekat dengan centroid
        sorted_nodes = sorted(alive_nodes, key=lambda node: -node['energy'])  # Urutkan berdasar energi
        cluster_heads = []
        
        for node in sorted_nodes:
            if len(cluster_heads) < num_ch:
                dist_to_avg = np.sqrt((node['x'] - avg_x)**2 + (node['y'] - avg_y)**2)
                # Cluster head dipilih berdasarkan energi tinggi dan jarak relatif ke centroid
                if dist_to_avg < 30:  # Threshold jarak ke centroid
                    cluster_heads.append(node)
        
        # Step 3: Simpan statistik
        total_alive = len(alive_nodes)
        total_dead = len(nodes) - total_alive
        stats['round'].append(r)
        stats['alive'].append(total_alive)
        stats['dead'].append(total_dead)
        stats['cluster_heads'].append(len(cluster_heads))

        # Step 4: Komunikasi dan update energi
        remaining_energy = sum([node['energy'] for node in nodes if node['energy'] > 0])
        
        # Kondisi jika tidak ada cluster head yang terpilih
        if not cluster_heads:
            for node in nodes:
                if node['energy'] > 0:
                    # Semua node langsung berkomunikasi ke sink
                    dist_to_sink = np.sqrt((node['x'] - sink['x'])**2 + (node['y'] - sink['y'])**2)
                    if dist_to_sink > do:
                        energy_consumed = (energy_params['E_tx'] * packet_size +
                                           energy_params['E_mp'] * packet_size * (dist_to_sink ** 4))
                    else:
                        energy_consumed = (energy_params['E_tx'] * packet_size +
                                           energy_params['E_fs'] * packet_size * (dist_to_sink ** 2))
                    node['energy'] -= energy_consumed
        else:
            # Energi untuk Cluster Head dan anggota cluster
            for node in nodes:
                if node in cluster_heads:
                    # Cluster head menerima pesan dari anggota cluster
                    for member in nodes:
                        if member != node and member['energy'] > 0:
                            dist_to_ch = np.sqrt((member['x'] - node['x'])**2 + (member['y'] - node['y'])**2)
                            # Biaya energi untuk anggota cluster yang mengirim data ke Cluster Head
                            if dist_to_ch > do:
                                energy_consumed = (energy_params['E_tx'] * packet_size +
                                                   energy_params['E_mp'] * packet_size * (dist_to_ch ** 4))
                            else:
                                energy_consumed = (energy_params['E_tx'] * packet_size +
                                                   energy_params['E_fs'] * packet_size * (dist_to_ch ** 2))
                            member['energy'] -= energy_consumed
                            node['energy'] -= energy_params['E_elec'] * packet_size  # Energi CH saat menerima pesan
                    # Cluster head mentransmisikan data agregat ke sink
                    dist_to_sink = np.sqrt((node['x'] - sink['x'])**2 + (node['y'] - sink['y'])**2)
                    if dist_to_sink > do:
                        energy_consumed = (energy_params['E_tx'] * packet_size +
                                           energy_params['E_mp'] * packet_size * (dist_to_sink ** 4))
                    else:
                        energy_consumed = (energy_params['E_tx'] * packet_size +
                                           energy_params['E_fs'] * packet_size * (dist_to_sink ** 2))
                    node['energy'] -= energy_consumed
                else:
                    # Anggota cluster mengirim data ke Cluster Head terdekat
                    distances = [np.sqrt((ch['x'] - node['x'])**2 + (ch['y'] - node['y'])**2) for ch in cluster_heads]
                    nearest_ch = cluster_heads[np.argmin(distances)]
                    nearest_ch_dist = min(distances)
                    if nearest_ch_dist > do:
                        energy_consumed = (energy_params['E_tx'] * packet_size +
                                           energy_params['E_mp'] * packet_size * (nearest_ch_dist ** 4))
                    else:
                        energy_consumed = (energy_params['E_tx'] * packet_size +
                                           energy_params['E_fs'] * packet_size * (nearest_ch_dist ** 2))
                    node['energy'] -= energy_consumed
                    # Cluster head menerima pesan dari node ini
                    nearest_ch['energy'] -= energy_params['E_elec'] * packet_size  # Energi CH saat menerima pesan

        # Step 5: Simpan sisa energi
        stats['energy'].append(sum([node['energy'] for node in nodes if node['energy'] > 0]))
    
    return stats
