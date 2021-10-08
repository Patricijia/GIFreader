interface ValidatorResponse {
    isValid: boolean,
    errorMessage: string
}

async function validateFileSize(fileSize: number): Promise<ValidatorResponse> {
    const fileSizeValidator = (await import('../validators/FileSizeValidator')).default

    const validator = new fileSizeValidator(fileSize);
    const isValid = validator.validateFileSize();

    return {
        isValid,
        errorMessage: isValid ? '' : validator.getErrorMessage()
    };
}

async function validateFileType(fileType: string): Promise<ValidatorResponse> {
    const fileTypeValidator = (await import('../validators/FileTypeValidator')).default;

    const validator = new fileTypeValidator(fileType);
    const isValid = validator.validate();

    return {
        isValid,
        errorMessage: isValid ? '' : validator.getErrorMessage()
    };
}

export {
    validateFileSize,
    validateFileType
}