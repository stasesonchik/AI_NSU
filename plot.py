# plot.py
import numpy as np
import matplotlib.pyplot as plt

# Тот же список классов:
list_of_heroes = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble',
                  'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler',
                  'comic_book_guy', 'disco_stu', 'edna_krabappel', 'fat_tony', 'gil', 'groundskeeper_willie',
                  'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lionel_hutz', 'lisa_simpson',
                  'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover',
                  'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner',
                  'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel',
                  'snake_jailbird', 'troy_mcclure', 'waylon_smithers']



# Загружаем
acc_matrix = np.load('batch128/acc_matrix.npy')  # shape (42, epochs)

# Готовим данные
final_acc = acc_matrix[:, -1]
n_classes = final_acc.shape[0]

# Строим bar-chart
plt.figure(figsize=(14,7))
plt.bar(range(n_classes), final_acc)
plt.xticks(range(n_classes), list_of_heroes, rotation=90)
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Per-Class Accuracy (Final Epoch)')
plt.tight_layout()
plt.savefig('class_accuracy_bar.png', dpi=300)
plt.close()
print("Bar chart saved as class_accuracy_bar.png")
