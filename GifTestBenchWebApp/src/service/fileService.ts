import GifDescription from "../model/GifDescription";

interface UploadFileResponse {
    success: boolean,
    description: GifDescription
}

class FileService 
{
    private file: File;

    constructor(file: File) {
        this.file = file;
    }

    static getFileExtension(fileName: string): string {
        const fileNames: Array<string> = fileName.split('.');

        if (fileNames.length === 0) {
            return '';
        }

        return fileNames[fileNames.length - 1];
    }

    async uploadFile(): Promise<UploadFileResponse> {
        const uploadResponse = await fetch('https://gifdescriptionsservice.azurewebsites.net/gif/upload', {
        // const uploadResponse = await fetch('https://localhost:5001/gif/upload', {
            method: 'POST',
            body: this.getFormData()
        });

        if(uploadResponse.status === 200)
        {
            const responseJson = await uploadResponse.json();
            return {
                success: uploadResponse.status === 200,
                description: responseJson
            };
        }
        else 
        {
            const uploadResponseRetry = await fetch('https://gifdescriptionsservice.azurewebsites.net/gif/upload', {
                // const uploadResponse = await fetch('https://localhost:5001/gif/upload', {
                    method: 'POST',
                    body: this.getFormData()
                });
            if(uploadResponseRetry.status === 200)
            {
                const responseJson = await uploadResponseRetry.json();
                return {
                    success: uploadResponseRetry.status === 200,
                    description: responseJson
                };
            }
            else
            {
                const uploadResponseRetryAgain = await fetch('https://gifdescriptionsservice.azurewebsites.net/gif/upload', {
                // const uploadResponse = await fetch('https://localhost:5001/gif/upload', {
                    method: 'POST',
                    body: this.getFormData()
                });
                const responseJson = await uploadResponseRetryAgain.json();
                return {
                    success: uploadResponseRetryAgain.status === 200,
                    description: responseJson
                };                
            }
        }        
    }

    async uploadFileEnhanced(): Promise<UploadFileResponse> {
        const uploadResponse = await fetch('https://gifdescriptorservice.azurewebsites.net/gif/upload', {
        // const uploadResponse = await fetch('https://localhost:5001/gif/upload', {
            method: 'POST',
            body: this.getFormData()
        });

        const responseJson = await uploadResponse.json();

        return {
            success: uploadResponse.status === 200,
            description: responseJson
        };
    }

    private getFormData(): FormData {
        const formData = new FormData();
        formData.append('file', this.file);        
        return formData;
    }
}

export default FileService;