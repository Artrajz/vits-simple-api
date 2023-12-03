const BackgroundUploader = {
    fetchBackgroundImage: function () {
        const savedBackgroundImageUrl = localStorage.getItem('backgroundImageUrl');
        if (savedBackgroundImageUrl) {
            document.body.style.backgroundImage = `url(${savedBackgroundImageUrl})`;
        }
    },

    uploadBackgroundImage: function (event) {
        const file = event.target.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                const backgroundImageUrl = e.target.result;

                document.body.style.backgroundImage = `url(${backgroundImageUrl})`;

                localStorage.setItem('backgroundImageUrl', backgroundImageUrl);

                BackgroundUploader.saveImageLocally(file);
            };

            reader.readAsDataURL(file);
        }
    },

    saveImageLocally: function (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            const imageData = e.target.result.split(',')[1];
            const blob = BackgroundUploader.base64ToBlob(imageData);
            const imageName = 'background.jpg';

            BackgroundUploader.deletePreviousImage(imageName).then(() => {
                BackgroundUploader.saveNewImage(imageName, blob);
            });
        };

        reader.readAsDataURL(file);
    },

    base64ToBlob: function (base64) {
        const byteString = atob(base64);
        const arrayBuffer = new ArrayBuffer(byteString.length);
        const intArray = new Uint8Array(arrayBuffer);

        for (let i = 0; i < byteString.length; i++) {
            intArray[i] = byteString.charCodeAt(i);
        }

        return new Blob([arrayBuffer], { type: 'image/jpeg' });
    },

    deletePreviousImage: function (imageName) {
        // 删除原有图片逻辑，这里简化为返回 Promise
        return new Promise((resolve, reject) => {
            resolve();
        });
    },

    saveNewImage: function (imageName, blob) {
        const imageUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = imageUrl;
        a.download = imageName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        URL.revokeObjectURL(imageUrl);
    }
};

document.addEventListener('DOMContentLoaded', function () {
    BackgroundUploader.fetchBackgroundImage();

    document.getElementById('uploadInput').addEventListener('change', function(event) {
        BackgroundUploader.uploadBackgroundImage(event);
    });
});
