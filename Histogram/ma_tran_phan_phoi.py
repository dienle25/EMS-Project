plt.figure(figsize=(12, 8))
# Vẽ boxplot nằm ngang cho tất cả các cột trừ cột exam_score
sns.boxplot(data=df.drop(columns=['exam_score']), orient='h', palette='Set2')
plt.title('So sánh độ trải rộng và Outliers của các biến', fontsize=15)
plt.savefig('all_features_boxplot.png')
plt.show()