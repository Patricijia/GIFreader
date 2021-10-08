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
        const uploadResponse = await fetch('https://localhost:5001/gif/upload', {
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
        formData.append("originalUri", "https://originalURL/");
        return formData;
    }
}

export default FileService;