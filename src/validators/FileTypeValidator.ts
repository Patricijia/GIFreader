class FileTypeValidator
{
    private fileType: string;
    private validTypes: string[] = ["gif", "mp4"];

    constructor(fileType: string) {
        this.fileType = fileType;        
    }

    validate() : boolean {
        return this.validTypes.includes(this.fileType);
    }

    getErrorMessage() : string {
        return `${this.fileType} is not an acceptable file type.`;
    }
}

export default FileTypeValidator;