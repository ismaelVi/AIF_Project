# 🎥 AIF Movie Project

Welcome to the **AIF Movie Project**, a web application powered by Gradio and run using Docker. 🚀

---

## 🛠️ **Execution Instructions**

### 1️⃣ **Build the Docker Image** 🐳
```bash
docker compose build
```

### 2️⃣ **Run the Container** ▶️
```bash
docker compose up
```

### 3️⃣ **Access the Web Application** 🌐
The application will be available at:  
🔗 [http://127.0.0.1:7860/](http://127.0.0.1:7860/)

### 4️⃣ **Test the API**
Once the application is running, you can test the system using the two images below:

📌 Image triggering an anomaly ( for genre prediction) 🛑
<br>
![test1](https://github.com/user-attachments/assets/88946dbc-ffa5-4f43-b354-eed2bebd9e52)


📌 Image triggering a valid movie result (can be use for genre prediction and recommandation)🎬 
<br>
![110](https://github.com/user-attachments/assets/6d34c32b-1ca9-479c-b331-efa0085ad3f1)


You can also test the recommandation with description by providing a movie description. 
<br>
Here is an example: "the adventures of Woody, a cowboy doll who feels threatened when his owner, Andy, receives a new, high-tech action figure—Buzz Lightyear"

*ℹ️ Note: The models are poorly trained, so the results may not be accurate.
For example, the anomaly detection model rarely triggers an anomaly case. This is why we provide an example to ensure it still works correctly.*

### 5️⃣ **Stop the Containers** 🛑
To stop the application, choose one of the following options:  
```bash
docker compose down
```
or press **`Ctrl+C`** in the terminal where the container is running.

---

## ⚠️ **If a "Permission Denied" Error Occurs:**

On Linux machines, you may need to run commands with `sudo`. 🔒

### 1️⃣ **Build the Docker Image with sudo**
```bash
sudo docker compose build
```

### 2️⃣ **Run the Container with sudo**
```bash
sudo docker compose up
```

### 3️⃣ **Access the Web Application** 🌐
The application will be available at:  
🔗 [http://127.0.0.1:7860/](http://127.0.0.1:7860/)

### 4️⃣ **Test the API**
Once the application is running, you can test the system using the two images below:

📌 Image triggering an anomaly ( for genre prediction) 🛑 
<br>
![test1](https://github.com/user-attachments/assets/88946dbc-ffa5-4f43-b354-eed2bebd9e52)


📌 Image triggering a valid movie result (can be use for genre prediction and recommandation)🎬
<br>
![110](https://github.com/user-attachments/assets/6d34c32b-1ca9-479c-b331-efa0085ad3f1)


You can also test the recommandation with description by providing a movie description.
<br>
Here is an example: "the adventures of Woody, a cowboy doll who feels threatened when his owner, Andy, receives a new, high-tech action figure—Buzz Lightyear"

*ℹ️ Note: The models are poorly trained, so the results may not be accurate.
For example, the anomaly detection model rarely triggers an anomaly case. This is why we provide an example to ensure it still works correctly.*

### 5️⃣ **Stop the Containers with sudo** 🛑
To stop the application, choose one of the following options:  
```bash
sudo docker compose down
```
or press **`Ctrl+C`** in the terminal where the container is running.

---

## 📌 **Important Notes:**
- ✅ **Prerequisites:** Ensure that Docker and Docker Compose are installed and configured on your machine.  
- 🛠️ **Troubleshooting:** If issues arise, check your permissions or refer to the official Docker documentation.  

---

## 🤝 **Contributions**
We welcome your contributions! Feel free to submit suggestions for improvements or report any issues. 🚀

---

✨ Thank you for using the **AIF Movie Project**! Have fun! 🎬
