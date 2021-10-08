class FileSizeValidator
{
    private fileSizeInBytes: number
    private maxFileSizeInBytes: number = 4597150

    constructor(fileSizeInBytes: number) {
        this.fileSizeInBytes = fileSizeInBytes;
    }

    validateFileSize(): boolean {
        return this.fileSizeInBytes <= this.maxFileSizeInBytes;
    }

    getErrorMessage(): string {
        return 'Max file size allowed is 4MB';
    }
}

export default FileSizeValidator;