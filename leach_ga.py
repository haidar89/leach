import numpy as np
from random import choices, random

# Fungsi fitness untuk mengevaluasi kualitas dari pemilihan cluster head
def fitness(cluster_heads, nodes, sink, energy_params):
    # Jika cluster_heads kosong, kembalikan nilai fitness yang rendah
    if not cluster_heads:
        return -np.inf  # Ini sebagai indikator solusi yang tidak valid
    
    # Maksimalkan energi tersisa di cluster heads
    energy_score = sum([node['energy'] for node in cluster_heads]) / len(cluster_heads)
    
    # Rata-rata jarak dari setiap cluster head ke sink
    distance_to_sink = [np.linalg.norm([ch['x'] - sink['x'], ch['y'] - sink['y']]) for ch in cluster_heads]
    mean_distance_to_sink = np.mean(distance_to_sink)
    
    # Rata-rata jarak antar cluster heads untuk menghindari lokasi terlalu berdekatan
    distance_between_ch = [np.linalg.norm([ch1['x'] - ch2['x'], ch1['y'] - ch2['y']]) 
                           for i, ch1 in enumerate(cluster_heads) 
                           for ch2 in cluster_heads[i+1:]]
    avg_distance_between_ch = np.mean(distance_between_ch) if distance_between_ch else 0
    
    # Parameter konstanta untuk mengatur kontribusi masing-masing komponen
    alpha, beta, gamma = 1, 0.01, 0.1
    return alpha * energy_score - beta * mean_distance_to_sink + gamma * avg_distance_between_ch

# Fungsi utama algoritma genetika untuk pemilihan cluster head
def leach_ga(nodes, sink, energy_params, packet_size, do, rounds, population_size=10, mutation_rate=0.1):
    # Variabel untuk menyimpan statistik setiap ronde
    stats = {
        'round': [], 'alive': [], 'dead': [], 'cluster_heads': [], 'energy': []
    }
    
    for r in range(rounds):
        # Inisialisasi populasi, setiap individu adalah kandidat cluster head
        population = []
        for _ in range(population_size):
            individual = choices(nodes, k=int(len(nodes) * energy_params['p']))  # Pilih subset acak nodes
            population.append(individual)

        # Evaluasi fitness untuk setiap individu dalam populasi
        fitness_scores = [fitness(individual, nodes, sink, energy_params) for individual in population]
        
        # Seleksi dua parent terbaik berdasarkan nilai fitness
        parent_indices = np.argsort(fitness_scores)[-2:]  # Ambil dua individu terbaik
        parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
        
        # Crossover: kombinasikan parent1 dan parent2 untuk membuat child
        crossover_point = len(parent1) // 2  # Tentukan titik crossover
        child = parent1[:crossover_point] + [node for node in parent2[crossover_point:] if node not in parent1]
        
        # Mutasi: tambahkan atau hapus node secara acak dalam cluster head child
        if random() < mutation_rate:
            if len(child) > 0:
                if random() < 0.5:
                    # Mutasi Hapus: hapus node acak dari child jika lebih dari satu cluster head
                    child.pop(np.random.randint(len(child)))
                else:
                    # Mutasi Tambah: tambahkan node baru acak ke child jika belum ada
                    potential_nodes = [node for node in nodes if node not in child]
                    if potential_nodes:
                        new_node = choices(potential_nodes, k=1)
                        child.extend(new_node)

        # Hasil individu terbaik dari populasi sebagai cluster head untuk ronde ini
        cluster_heads = child
        remaining_energy = sum([node['energy'] for node in nodes if node['energy'] > 0])

        # Update energi nodes berdasarkan jarak ke cluster head atau sink
        for node in nodes:
            if node['energy'] > 0:
                if node in cluster_heads:
                    # Cluster head mengirim langsung ke sink
                    distance_to_sink = np.linalg.norm([node['x'] - sink['x'], node['y'] - sink['y']])
                    if distance_to_sink > do:
                        energy_cost = (energy_params['E_tx'] * packet_size + energy_params['E_mp'] * packet_size * (distance_to_sink ** 4))
                    else:
                        energy_cost = (energy_params['E_tx'] * packet_size + energy_params['E_fs'] * packet_size * (distance_to_sink ** 2))
                    node['energy'] = max(node['energy'] - energy_cost, 0)
                else:
                    # Node biasa mengirim data ke cluster head terdekat
                    distances = [np.linalg.norm([ch['x'] - node['x'], ch['y'] - node['y']]) for ch in cluster_heads]
                    if distances:
                        nearest_ch_dist = min(distances)
                        if nearest_ch_dist > do:
                            energy_cost = energy_params['E_tx'] * packet_size + energy_params['E_mp'] * packet_size * (nearest_ch_dist ** 4)
                        else:
                            energy_cost = energy_params['E_tx'] * packet_size + energy_params['E_fs'] * packet_size * (nearest_ch_dist ** 2)
                        node['energy'] = max(node['energy'] - energy_cost, 0)

                        # Cluster head menerima data dari node biasa
                        nearest_ch = cluster_heads[distances.index(nearest_ch_dist)]
                        nearest_ch['energy'] = max(nearest_ch['energy'] - energy_params['E_elec'] * packet_size, 0)

        # Simpan statistik untuk analisis
        alive_nodes = sum([1 for node in nodes if node['energy'] > 0])
        dead_nodes = len(nodes) - alive_nodes
        stats['round'].append(r)
        stats['alive'].append(alive_nodes)
        stats['dead'].append(dead_nodes)
        stats['cluster_heads'].append(len(cluster_heads))
        stats['energy'].append(remaining_energy)
    
    return stats
