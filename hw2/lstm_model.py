import tensorflow as tf
import numpy as np


class LSTMConfig:
    max_document_length = 600
    num_class = 10
    embedding_size = 64
    lstm_size_each_layer = '256,128'
    use_bidirectional = False
    use_basic_cell = 1
    use_attention = True
    attention_size = 200
    grad_clip = 5.0
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32
    dropout_keep_prob = 0.5
    save_per_batch = 10
    print_per_batch = 128
    vocab_size = 5000  # 假设词汇表大小


class LSTMModel(tf.keras.Model):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config

        # Embedding层
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.embedding_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            name="embedding"
        )

        # LSTM层
        lstm_sizes = [int(size.strip()) for size in self.config.lstm_size_each_layer.split(',')]
        self.lstm_layers = [
            tf.keras.layers.LSTM(size, return_sequences=True, dropout=self.config.dropout_keep_prob)
            for size in lstm_sizes
        ]
        if self.config.use_bidirectional:
            self.lstm_layers = [
                tf.keras.layers.Bidirectional(layer) for layer in self.lstm_layers
            ]

        # Attention 层
        self.use_attention = self.config.use_attention
        if self.use_attention:
            self.attention_w = tf.keras.layers.Dense(self.config.attention_size, activation='tanh')
            self.attention_u = tf.keras.layers.Dense(1, activation=None)

        # F全连接layer
        self.fc = tf.keras.layers.Dense(self.config.num_class)

    def call(self, inputs, training=None):
        x, y = inputs

        # Embedding
        x = self.embedding(x)

        # LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)

        # Attention mechanism
        if self.use_attention:
            u = self.attention_w(x)  # Apply attention weights
            attention_scores = tf.nn.softmax(self.attention_u(u), axis=1)
            x = tf.reduce_sum(x * attention_scores, axis=1)
        else:
            x = x[:, -1, :]  # 使用上一个时间步的输出

        # 全连接层
        logits = self.fc(x)
        predictions = tf.nn.softmax(logits, axis=1)

        return logits, predictions


class LSTMTrainer:
    def __init__(self, config):
        self.config = config
        self.model = LSTMModel(config)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits, _ = self.model((x_batch, y_batch), training=True)
            loss = self.loss_fn(y_batch, logits)

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def evaluate(self, x, y):
        logits, predictions = self.model((x, y), training=False)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(y, axis=1)), dtype=tf.float32)
        )
        return accuracy



# config = LSTMConfig()
# trainer = LSTMTrainer(config)

# # Fake data for demonstration purposes
# x_train = np.random.randint(0, config.vocab_size, size=(100, config.max_document_length))
# y_train = tf.keras.utils.to_categorical(np.random.randint(0, config.num_class, size=(100,)), num_classes=config.num_class)

#训练
for epoch in range(config.num_epochs):
    batch_loss = trainer.train_step(x_train, y_train)
    print(f"Epoch {epoch + 1}, Loss: {batch_loss.numpy()}")

# 评估
accuracy = trainer.evaluate(x_train, y_train)
print(f"Training Accuracy: {accuracy.numpy()}")
# import tensorflow as tf
# import numpy as np


# class LSTMConfig:
#     max_document_length = 600
#     num_class = 10
#     embedding_size = 64
#     lstm_size_each_layer = '256,128'
#     use_bidirectional = False
#     use_basic_cell = 1
#     use_attention = True
#     attention_size = 200
#     grad_clip = 5.0
#     learning_rate = 0.001
#     num_epochs = 10
#     batch_size = 32
#     dropout_keep_prob = 0.5
#     save_per_batch = 10
#     print_per_batch = 128
#     vocab_size = 5000  # 假设词汇表大小


# class LSTMModel(tf.keras.Model):
#     def __init__(self, config):
#         super(LSTMModel, self).__init__()
#         self.config = config

#         # Embedding layer
#         self.embedding = tf.keras.layers.Embedding(
#             input_dim=self.config.vocab_size,
#             output_dim=self.config.embedding_size,
#             embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             name="embedding"
#         )

#         # LSTM Layers
#         lstm_sizes = [int(size.strip()) for size in self.config.lstm_size_each_layer.split(',')]
#         self.lstm_layers = [
#             tf.keras.layers.LSTM(size, return_sequences=True, dropout=self.config.dropout_keep_prob)
#             for size in lstm_sizes
#         ]
#         if self.config.use_bidirectional:
#             self.lstm_layers = [
#                 tf.keras.layers.Bidirectional(layer) for layer in self.lstm_layers
#             ]

#         # Attention layer
#         self.use_attention = self.config.use_attention
#         if self.use_attention:
#             self.attention_w = tf.keras.layers.Dense(self.config.attention_size, activation='tanh')
#             self.attention_u = tf.keras.layers.Dense(1, activation=None)

#         # Fully connected layer
#         self.fc = tf.keras.layers.Dense(self.config.num_class)

#     def call(self, inputs, training=None):
#         x, y = inputs

#         # Embedding
#         x = self.embedding(x)

#         # LSTM layers
#         for lstm_layer in self.lstm_layers:
#             x = lstm_layer(x)

#         # Attention mechanism
#         if self.use_attention:
#             u = self.attention_w(x)  # Apply attention weights
#             attention_scores = tf.nn.softmax(self.attention_u(u), axis=1)
#             x = tf.reduce_sum(x * attention_scores, axis=1)
#         else:
#             x = x[:, -1, :]  # Use the output of the last time step

#         # Fully connected layer
#         logits = self.fc(x)
#         predictions = tf.nn.softmax(logits, axis=1)

#         return logits, predictions


# class LSTMTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.model = LSTMModel(config)
#         self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

#     def train_step(self, x_batch, y_batch):
#         with tf.GradientTape() as tape:
#             logits, _ = self.model((x_batch, y_batch), training=True)
#             loss = self.loss_fn(y_batch, logits)

#         # Compute and apply gradients
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#         return loss

#     def evaluate(self, x, y):
#         logits, predictions = self.model((x, y), training=False)
#         accuracy = tf.reduce_mean(
#             tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(y, axis=1)), dtype=tf.float32)
#         )
#         return accuracy


# # Example usage
# config = LSTMConfig()
# trainer = LSTMTrainer(config)

# # Fake data for demonstration purposes
# x_train = np.random.randint(0, config.vocab_size, size=(100, config.max_document_length))
# y_train = tf.keras.utils.to_categorical(np.random.randint(0, config.num_class, size=(100,)), num_classes=config.num_class)

# # Training
# for epoch in range(config.num_epochs):
#     batch_loss = trainer.train_step(x_train, y_train)
#     print(f"Epoch {epoch + 1}, Loss: {batch_loss.numpy()}")

# # Evaluation
# accuracy = trainer.evaluate(x_train, y_train)
# print(f"Training Accuracy: {accuracy.numpy()}")
